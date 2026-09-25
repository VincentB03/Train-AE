#!/usr/bin/env python
r"""Residual diagnostics of trained galaxy autoencoders, on the test split.

Compares several checkpoints on the same test images, with n = (x - y) / rms:

  1. Noise floor: on background pixels (every model predicts ~0) the residual is
     pure noise. Its spread gives c = true sigma / noise_map, and its student-t
     NLL the lowest reachable loss. excess = loss - floor is what is left to gain.
  2. Residuals binned by predicted SNR: where each model gains or loses. The
     excess is split into a (Poisson noise missing from noise_map, a property of
     the data) and f (fractional flux error, the part a better model can remove).
  3. Latent usage: how many latent dimensions carry the variance.

"loss/image" must match the loss_test logged at that epoch; if not, the split,
epoch or checkpoint is wrong.

Run from the repository root, on one GPU. Runs are directory names under
PATH/runs; the first one is the reference:

    python -m experiments.evaluate_residuals --epoch 1000 <run_a> <run_b>
"""
import argparse

import numpy as np
import jax
import equinox as eqx
from datasets import load_dataset

from pshear.utils import load_galaxy_autoencoder
from experiments.utils import PATH

DATASET_NAME = "VincentB03/Euclid-Q1-postage-stamps"

# loss_test only covers the first (N // 512) * 512 test images (sequential
# loader, drop_last): use the same ones
TEST_BATCH = 512
EVAL_BATCH = 128       # forward-pass batch: memory only, no effect on results
NU = 5.0               # GalaxyAutoEncoderLoss default
EPS = 1e-8             # same eps as the loss
BG_SNR = 0.1           # background: every model predicts less than 0.1 sigma
SNR_EDGES = [0, 0.1, 0.3, 1, 3, 10, 30, 100, 300, np.inf]


def column(batch, name, dtype):
    # (N, 1, H, W), like the training batches (see as_batch in train_partial_parallel.py)
    col = np.asarray(batch[name])
    col = np.stack(col) if col.dtype == object else col
    return col.astype(dtype, copy=False)[:, None]


@eqx.filter_jit
def forward(model, x, psf):
    y, _, z = jax.vmap(model)(x, psf)
    return y, z


def predict(model, x, psf):
    ys, zs = [], []
    for i in range(0, len(x), EVAL_BATCH):
        y, z = forward(model, x[i:i + EVAL_BATCH], psf[i:i + EVAL_BATCH])
        ys.append(np.asarray(y))
        zs.append(np.asarray(z))
    return np.concatenate(ys).astype(np.float64), np.concatenate(zs)


def student_t_nll(r, sigma2):
    return (NU + 1) / 2 * np.log1p(r ** 2 / (NU * sigma2))


def per_image_mean(v, m):
    # same reduction as the loss: masked mean per image, then mean over images
    return ((v * m).sum(axis=(1, 2, 3)) / (m.sum(axis=(1, 2, 3)) + EPS)).mean()


def poisson_and_flux_error(n, snr, mask, c2):
    # fit (mean(n^2) - c^2) / <SNR> = a + f^2 * <SNR^2>/<SNR> over the bins with
    # SNR >= 1: a = Poisson noise missing from noise_map (data), f = fractional
    # flux error (model)
    xs, ys = [], []
    for lo, hi in zip(SNR_EDGES[:-1], SNR_EDGES[1:]):
        sel = mask & (snr >= max(lo, 1)) & (snr < hi)
        if lo < 1 or not sel.any():
            continue
        s = snr[sel]
        xs.append((s ** 2).mean() / s.mean())
        ys.append(((n[sel] ** 2).mean() - c2) / s.mean())
    f2, a = np.polyfit(xs, ys, 1)
    return a, np.sqrt(max(f2, 0.0))


def latent_usage(z):
    flat = z.reshape(len(z), -1)
    eig = np.linalg.eigvalsh(np.cov(flat, rowvar=False))[::-1].clip(min=0)
    cum = np.cumsum(eig) / eig.sum()
    std = flat.std(axis=0)
    per_channel = z.var(axis=0).reshape(z.shape[1], -1).sum(axis=1)
    return {
        "dims": flat.shape[1],
        "d95": int(np.searchsorted(cum, 0.95) + 1),
        "d99": int(np.searchsorted(cum, 0.99) + 1),
        "participation": eig.sum() ** 2 / (eig ** 2).sum(),
        "dead": int((std < 0.01 * std.max()).sum()),
        "channel_share": per_channel / per_channel.sum(),
    }


parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("runs", nargs="+",
                    help="run directories under PATH/runs; the first is the reference")
parser.add_argument("--epoch", type=int, default=1000,
                    help="checkpoint epoch, the same for every run (default: 1000)")
args = parser.parse_args()
RUNS, EPOCH = args.runs, args.epoch

# 1) the test images, SAME split as training (seed=42)
print(f"Loading {DATASET_NAME}")
dset = load_dataset(DATASET_NAME, split="train", keep_in_memory=True)
dset_test = dset.train_test_split(test_size=5000, seed=42)["test"].with_format("numpy")
n_eval = (len(dset_test) // TEST_BATCH) * TEST_BATCH
batch = dset_test.select_columns(
    ["sci_subtracted", "psf_residual", "noise_map", "binary_mask"]
)[:n_eval]
x = column(batch, "sci_subtracted", np.float32)
psf = column(batch, "psf_residual", np.float32)
rms = column(batch, "noise_map", np.float64)
mask = column(batch, "binary_mask", bool)
sigma2 = rms ** 2 + EPS
sigma = np.sqrt(sigma2)
x64 = x.astype(np.float64)
print(f"{n_eval} test images (of {len(dset_test)}), {int(mask.sum())} unmasked pixels")

# 2) every model on the same images
results = {}
for run in RUNS:
    print(f"Evaluating {run} at epoch {EPOCH}")
    model = load_galaxy_autoencoder(PATH / "runs" / run, epoch=EPOCH)
    model = eqx.nn.inference_mode(model, value=True)
    y, z = predict(model, x, psf)
    r = x64 - y
    results[run] = {"y": y, "n": r / sigma, "nll": student_t_nll(r, sigma2),
                    "chi2": r ** 2 / sigma2, "z": z}

# 3) pixels binned on the models' mean prediction, so all models are compared
# on the same pixels
snr = np.mean([np.abs(res["y"]) for res in results.values()], axis=0) / sigma
bg = mask & (snr < BG_SNR)
floors = {run: res["nll"][bg].mean() for run, res in results.items()}
# the model with the cleanest background sets the floor
floor = min(floors.values())
n_bg = results[RUNS[0]]["n"][bg]
c_std = n_bg.std()
c_mad = 1.4826 * np.median(np.abs(n_bg - np.median(n_bg)))

print("\n=== Noise floor, from the background ===")
print(f"background pixels: {bg.sum() / mask.sum():.1%} of the unmasked ones "
      f"(|y| < {BG_SNR} sigma for every model)")
print(f"c = true sigma / noise_map:  {c_std:.4f} (std)   {c_mad:.4f} (MAD, robust)")
for run in RUNS:
    print(f"  background NLL, {run:34s} {floors[run]:.5f}")
print(f"floor (lowest of the above): {floor:.5f}")
print("  -> the models should agree to ~1e-4 here; a clearly higher value means "
      "that model paints structure on empty sky")

print("\n=== Global metrics ===")
print(f"{'run':36s} {'loss/image':>10s} {'loss/pixel':>10s} {'excess':>8s} "
      f"{'chi2':>7s} {'MSE':>9s}")
for run, res in results.items():
    loss_img = per_image_mean(res["nll"], mask)
    loss_pix = res["nll"][mask].mean()
    print(f"{run:36s} {loss_img:10.5f} {loss_pix:10.5f} {loss_pix - floor:+8.5f} "
          f"{res['chi2'][mask].mean():7.4f} {((x64 - res['y']) ** 2)[mask].mean():9.3f}")
print("  loss/image must match the loss_test logged at this epoch (~1e-4).")
print(f"  excess = loss/pixel - floor: what is left to gain. chi2 would be "
      f"{c_std ** 2:.4f} (= c^2) for a perfect model.")

print("\n=== By predicted SNR (same pixels for every model) ===")
for run, res in results.items():
    print(f"\n{run}")
    print(f"{'SNR bin':>13s} {'pixels':>7s} {'std(n)':>7s} {'mean(n)':>8s} "
          f"{'NLL':>7s} {'share of excess':>15s} {'excess m2/SNR':>14s}")
    excess_total = res["nll"][mask].mean() - floor
    for lo, hi in zip(SNR_EDGES[:-1], SNR_EDGES[1:]):
        sel = mask & (snr >= lo) & (snr < hi)
        if not sel.any():
            continue
        n_bin = res["n"][sel]
        frac = sel.sum() / mask.sum()
        share = frac * (res["nll"][sel].mean() - floor) / excess_total
        excess_m2 = (n_bin ** 2).mean() - c_std ** 2
        print(f"{lo:5g}-{hi:<7g} {frac:7.2%} {n_bin.std():7.3f} {n_bin.mean():+8.3f} "
              f"{res['nll'][sel].mean():7.4f} {share:15.1%} "
              f"{excess_m2 / snr[sel].mean():14.4f}")

print("\n=== What the excess is made of (fit over the bins with SNR >= 1) ===")
print(f"{'run':36s} {'a (Poisson, data)':>18s} {'f (flux error, model)':>22s}")
for run, res in results.items():
    a, f = poisson_and_flux_error(res["n"], snr, mask, c_std ** 2)
    print(f"{run:36s} {a:18.4f} {f:21.1%}")
print("  a is a property of the data: it must come out about the same for every model,")
print("  otherwise the two-term model does not describe these residuals -- read the")
print("  per-bin tables instead. f is what a better model can still remove; a is not.")

if len(RUNS) > 1:
    base = RUNS[0]
    print(f"\n=== std(n) difference vs {base} (negative = better) ===")
    print(f"{'SNR bin':>13s} " + " ".join(f"{run[:22]:>22s}" for run in RUNS[1:]))
    for lo, hi in zip(SNR_EDGES[:-1], SNR_EDGES[1:]):
        sel = mask & (snr >= lo) & (snr < hi)
        if not sel.any():
            continue
        ref = results[base]["n"][sel].std()
        diffs = [results[run]["n"][sel].std() - ref for run in RUNS[1:]]
        print(f"{lo:5g}-{hi:<7g} " + " ".join(f"{d:+22.4f}" for d in diffs))

print("\n=== Latent usage ===")
print(f"{'run':36s} {'dims':>5s} {'95% var':>8s} {'99% var':>8s} "
      f"{'particip.':>9s} {'dead':>5s}  variance share per channel")
for run, res in results.items():
    u = latent_usage(res["z"])
    shares = "  ".join(f"{s:.1%}" for s in u["channel_share"])
    print(f"{run:36s} {u['dims']:5d} {u['d95']:8d} {u['d99']:8d} "
          f"{u['participation']:9.1f} {u['dead']:5d}  {shares}")
