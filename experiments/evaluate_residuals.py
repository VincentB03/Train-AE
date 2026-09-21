#!/usr/bin/env python
r"""Residual diagnostics of trained galaxy autoencoders, on the test split.

Answers three questions without any training, by comparing several checkpoints
on exactly the same test images:

  1. Where is the noise floor, and how far above it is each model?
  2. Where, in signal-to-noise, does one model beat another?
  3. How many of its latent dimensions does each model actually use?

Why a measurement is needed
---------------------------
If the reconstruction were perfect, the residual x - y would be pure noise and
the student-t loss would sit at a floor that depends only on how well noise_map
describes the real noise. With c = true sigma / noise_map, that floor is 0.483
for c = 1.00 but 0.524 for c = 1.05: a 1% error on c moves the floor by about as
much as everything the model has gained so far. Until c is measured, "how much is
left to gain" has no answer.

How
---
Most pixels of a 64x64 stamp are empty sky. There the model has nothing to
reconstruct and predicts ~0, so the residual is the noise itself: the BACKGROUND
calibrates noise_map, independently of the model. The SOURCE pixels then measure
the model. Concretely, on the unmasked pixels, n = (x - y) / rms, and:

  - background = pixels where every model predicts |y| < BG_SNR * rms. Their mean
    student-t NLL is the floor, measured directly (no Gaussian assumption); the
    spread of n there is c.
  - every pixel is binned by predicted SNR, |y| / rms, averaged over the models so
    that all models are binned on the SAME pixels. Per bin: spread of n, its mean
    (flux bias), and the share of the excess loss that bin carries.

Reading it
----------
  - "loss (per image)" must reproduce the loss_test logged at that epoch to ~1e-4.
    If not, the split, the epoch or the checkpoint is wrong -- trust nothing else.
  - excess = loss - floor is what is left to gain. Its per-bin split says WHERE.
  - Comparing two models bin by bin is immune to a wrong noise_map: both are
    measured against the same one, so their difference is model error only.
  - Caveat, and it is a big one: if noise_map holds only the sky noise, it misses
    the source's own Poisson noise, and the bright pixels look bad even for a
    PERFECT model -- "excess" then overstates what is left to gain. On synthetic
    data with a perfect model and Poisson noise the size of the sky noise at
    SNR ~30, excess came out at +0.12. The two causes grow differently with SNR:
        mean(n^2) - c^2  =  a * SNR  +  f^2 * SNR^2
    a * SNR is Poisson noise noise_map left out (a property of the DATA, so it
    must come out the same for every model); f * SNR is a fractional flux error
    (the MODEL's, and the part a better model can remove). Dividing by SNR makes
    it a straight line in SNR, fitted over the bins with signal: intercept a,
    slope f^2. The "excess m2 / SNR" column is that line, bin by bin -- flat
    means Poisson only, rising means model error.
  - Latent: "dims for 99% var" is the dimensionality the model really uses, and
    the one the flow has to model. A 2-channel latent whose second channel
    carries ~0% of the variance did not use its extra room.

Run it from the repository root, on one GPU (inference only). Runs are directory
names under $SCRATCH/pshear/cosmos/runs; the first one is the reference the others
are compared against:

    srun python -m experiments.evaluate_residuals --epoch 1000 \
        Student-3-lr1.5e-4_<id> Student-4-latent2_<id>
"""
import argparse

import numpy as np
import jax
import equinox as eqx
from datasets import load_dataset

from pshear.utils import load_galaxy_autoencoder
from experiments.utils import PATH

DATASET_NAME = "VincentB03/euclid-Q1-VF"

# The training test loader is sequential with drop_last=True and batch 512, so
# loss_test covers only the first (N // 512) * 512 test images. Using the same
# ones makes the reproduction check exact.
TEST_BATCH = 512
EVAL_BATCH = 128       # forward-pass batch: memory only, no effect on results
NU = 5.0               # GalaxyAutoEncoderLoss default
EPS = 1e-8             # same eps as the loss
BG_SNR = 0.1           # background: every model predicts less than 0.1 sigma
SNR_EDGES = [0, 0.1, 0.3, 1, 3, 10, 30, 100, 300, np.inf]


def column(batch, name, dtype):
    # with_format("numpy") gives one (N, H, W) array when all rows share a
    # shape, otherwise an object array of (H, W) arrays (see as_batch in
    # train_partial_parallel.py); (N, 1, H, W) like the training batches
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
    # exactly the loss's reduction: masked mean per image, then mean over images
    return ((v * m).sum(axis=(1, 2, 3)) / (m.sum(axis=(1, 2, 3)) + EPS)).mean()


def poisson_and_flux_error(n, snr, mask, c2):
    # fit (mean(n^2) - c^2) / <SNR> = a + f^2 * <SNR^2>/<SNR> over the bins with
    # signal (SNR >= 1): a = Poisson noise missing from noise_map (data), f = an
    # effective fractional flux error (model). Second moment, not variance, so
    # that a biased model (e.g. a flux scale off by 5%) counts as an error too.
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


# runs on the command line rather than as constants here: nothing to edit on the
# cluster, so the checkout stays clean for the next git pull
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
dset_test = dset.train_test_split(test_size=0.1, seed=42)["test"].with_format("numpy")
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

# 3) shared pixel classes: binned on the models' MEAN prediction, so every model
# is judged on exactly the same pixels
snr = np.mean([np.abs(res["y"]) for res in results.values()], axis=0) / sigma
bg = mask & (snr < BG_SNR)
floors = {run: res["nll"][bg].mean() for run, res in results.items()}
# the model with the cleanest background sets the floor: a model that paints
# spurious structure on empty sky can only raise its own background NLL
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
