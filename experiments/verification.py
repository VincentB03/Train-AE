#!/usr/bin/env python
"""PQMass check that the flow generates galaxies from the same distribution as the
real data, at four points of the pipeline:

    Test 0 image   real vs real, raw pixels          -- calibration, no model
    Test 0 latent  real vs real, AE latents          -- calibration, encoder only
    Test 1         flow samples vs encoded real      -- the flow alone
    Test 2         full pipeline vs real, raw pixels -- flow + decoder + PSF

Each test draws N_SPLITS independent (x, y) pairs and keeps one p-value per pair;
under H0 these p-values are uniform on [0, 1]. (re_tessellation=k is not used: it
keeps x and y fixed, so its p-values are not independent.)

Figures are written to PQM_results/.
"""
from pathlib import Path

import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2, kstest

from datasets import load_dataset
from pqm import pqm_pvalue
from tqdm import tqdm

from pshear.utils import load_galaxy_autoencoder, load_flow, fetch_wandb_checkpoint

# relative to the repo root, not $SCRATCH (unlike experiments.utils.PATH)
ROOT = Path(".")

RESULTS_DIR = ROOT / "PQM_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# --- adapt to your run ---
WANDB_ENTITY = "vincentb03-imt-atlantique"
AE_RUN_PATH = f"{WANDB_ENTITY}/AE-partial-droppedDB/i344nq38"
AE_EPOCH = 2000
FLOW_RUN_PATH = f"{WANDB_ENTITY}/pshear-euclid-flow-dropped-db/9i28jqsm"
FLOW_EPOCH = 500
DATASET_NAME = "VincentB03/euclid-Q1-postage-stamps"

N_EVAL = 2000        # samples per side of one PQMass comparison
N_SPLITS = 200       # (x, y) pairs per test; one p-value each
NUM_REFS = 100
DOF = NUM_REFS - 1   # pqm rescales its chi2 to always use this dof
SPLIT_SEED = 20250909
ENCODE_BATCH = 500   # batch size for the one-off encode/decode/convolve passes

# Pool for the calibration tests:
#   "test" -> the 5000 test images, same pool as tests 1-2;
#   "all"  -> the whole dataset: splits barely overlap, but needs ~4 GB of RAM
#             and one encoder pass over every image.
CALIB_POOL = "all"

# pqm draws its reference points from the global numpy RNG
np.random.seed(0)

# downloaded to ROOT/wandb_weights/<run_id>/epoch_<epoch>/, or read from there if cached
AE_MODEL_PATH = fetch_wandb_checkpoint(AE_RUN_PATH, AE_EPOCH, cache_dir=ROOT / "wandb_weights")
FLOW_MODEL_PATH = fetch_wandb_checkpoint(FLOW_RUN_PATH, FLOW_EPOCH, cache_dir=ROOT / "wandb_weights")


def batched_vmap(fn, *arrays, batch=ENCODE_BATCH):
    """vmap ``fn`` over the leading axis in chunks, to bound device memory."""
    vfn = jax.vmap(fn)
    n = arrays[0].shape[0]
    out = [np.asarray(vfn(*(a[i:i + batch] for a in arrays))) for i in range(0, n, batch)]
    return np.concatenate(out, axis=0)


def pqm_split_pvalues(draw_pair, desc, n_splits=N_SPLITS, num_refs=NUM_REFS):
    """One p-value for each of ``n_splits`` fresh (x, y) pairs from ``draw_pair(rng)``.

    Reseeded from SPLIT_SEED, so every test uses the same splits. chi2 values are
    derived later with chi2.isf(pvals, DOF): calling pqm_chi2 too would double the cost.
    """
    rng = np.random.default_rng(SPLIT_SEED)
    pvals = np.empty(n_splits)
    for i in tqdm(range(n_splits), desc=desc):
        x, y = draw_pair(rng)
        pvals[i] = pqm_pvalue(x, y, num_refs=num_refs, z_score_norm=True)
    return pvals


def report_split_pvalues(label, pvals):
    """One summary line. Under H0: mean 0.5, frac(p<0.05) 0.05, KS not rejecting,
    median chi2/DoF ~ 1."""
    ks = kstest(pvals, "uniform")
    chi2_vals = chi2.isf(pvals, DOF)
    finite = chi2_vals[np.isfinite(chi2_vals)]
    med = f"{np.median(finite) / DOF:.3f}" if finite.size else "n/a"
    n_inf = chi2_vals.size - finite.size
    print(
        f"{label:<14s} n={pvals.size}  mean={pvals.mean():.3f}  "
        f"frac(p<0.05)={np.mean(pvals < 0.05):.3f}  "
        f"KS D={ks.statistic:.3f} p={ks.pvalue:.3g}  "
        f"median chi2/DoF={med}" + (f"  [{n_inf} p==0]" if n_inf else "")
    )
    return ks


def plot_chi2(slug, title, pipeline, pvals, n_bins=20):
    """Histogram of the PQM chi2 against the chi2(DOF) pdf (same p-values as
    plot_split_uniformity). p == 0 gives chi2 = inf: dropped and counted in the title."""
    chi2_vals = chi2.isf(np.asarray(pvals, dtype=float), DOF)
    finite = chi2_vals[np.isfinite(chi2_vals)]
    n_inf = chi2_vals.size - finite.size

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    if finite.size:
        lo = min(chi2.ppf(0.001, DOF), np.percentile(finite, 1))
        hi = max(chi2.ppf(0.999, DOF), np.percentile(finite, 99))
        n_out = int(np.sum((finite < lo) | (finite > hi)))
        ax.hist(finite, bins=n_bins, range=(lo, hi), density=True,
                edgecolor="black", linewidth=0.5, label=r"PQM $\chi^2$")
        grid = np.linspace(lo, hi, 400)
        ax.plot(grid, chi2.pdf(grid, df=DOF), color="red",
                label=rf"H0: $\chi^2$({DOF})")
        ax.axvline(DOF, color="red", ls=":", lw=1)
        ax.legend(fontsize=8)
    else:
        n_out = 0
        ax.text(0.5, 0.5, "all chi2 values non-finite (every p-value is 0)",
                ha="center", va="center", transform=ax.transAxes)

    notes = []
    if n_inf:
        notes.append(f"{n_inf}/{chi2_vals.size} non-finite (p==0) dropped")
    if n_out:
        notes.append(f"{n_out} outside the plotted range")
    tag = ("\n" + "; ".join(notes)) if notes else ""

    ax.set_xlabel(r"$\chi^2_{\rm PQM}$")
    ax.set_ylabel("density")
    ax.set_title(f"{title}\n{pipeline}{tag}", fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"{slug}_chi2.png", dpi=150)
    plt.close(fig)


def plot_split_uniformity(slug, title, pipeline, pvals, n_bins=10):
    """Left: p-value histogram with binomial bands. Right: empirical CDF with the KS 95%
    band."""
    pvals = np.asarray(pvals, dtype=float)
    n = pvals.size
    ks = kstest(pvals, "uniform")

    fig, (ax_h, ax_c) = plt.subplots(1, 2, figsize=(11, 4.6))

    counts, edges = np.histogram(pvals, bins=n_bins, range=(0, 1))
    expected = n / n_bins
    sigma = np.sqrt(n * (1 / n_bins) * (1 - 1 / n_bins))
    ax_h.bar(edges[:-1], counts, width=np.diff(edges), align="edge",
             edgecolor="black", linewidth=0.5)
    for k, alpha in ((2, 0.15), (1, 0.25)):
        ax_h.axhspan(expected - k * sigma, expected + k * sigma, color="red", alpha=alpha,
                     zorder=0, label=f"$\\pm{k}\\sigma$ binomial")
    ax_h.axhline(expected, color="red", label="H0: uniform")
    ax_h.set_xlim(0, 1)
    ax_h.set_xlabel("p-value")
    ax_h.set_ylabel(f"counts / {n} splits")
    ax_h.legend(fontsize=7, loc="lower left")

    ax_c.step(np.concatenate([[0], np.sort(pvals), [1]]),
              np.concatenate([[0], np.arange(1, n + 1) / n, [1]]), where="post")
    grid = np.linspace(0, 1, 200)
    band = 1.36 / np.sqrt(n)
    ax_c.plot(grid, grid, color="red", label="H0: uniform")
    ax_c.fill_between(grid, np.clip(grid - band, 0, 1), np.clip(grid + band, 0, 1),
                      color="red", alpha=0.15, label="KS 95% band")
    ax_c.set_xlim(0, 1)
    ax_c.set_ylim(0, 1)
    ax_c.set_xlabel("p-value")
    ax_c.set_ylabel("empirical CDF")
    ax_c.legend(fontsize=7, loc="lower right")

    fig.suptitle(
        f"{title}\n{pipeline}\n"
        f"{n} splits, 1 p-value each  |  mean={pvals.mean():.3f}  "
        f"frac(p<0.05)={np.mean(pvals < 0.05):.3f}  KS D={ks.statistic:.3f} p={ks.pvalue:.3g}",
        fontsize=9, y=0.995, va="top",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(RESULTS_DIR / f"{slug}_pvalue.png", dpi=150)
    plt.close(fig)


key = jax.random.key(0)

# 1) frozen models
ae = load_galaxy_autoencoder(AE_MODEL_PATH, epoch=AE_EPOCH)
ae = eqx.nn.inference_mode(ae, value=True)

flow = load_flow(FLOW_MODEL_PATH, epoch=FLOW_EPOCH)
flow = eqx.nn.inference_mode(flow, value=True)

# 2) real data: same test split as train_flow.py, so the flow is never tested on
# latents it was trained on
dset_full = load_dataset(DATASET_NAME, split="train", keep_in_memory=True)
dset = dset_full.train_test_split(test_size=5000, seed=42)
dset_test = dset["test"].with_format("numpy")

# 3) pools, computed once
test_imgs = np.asarray(dset_test["sci_subtracted"], dtype=np.float32)
test_psf = np.asarray(dset_test["psf_residual"], dtype=np.float32)
n_test = test_imgs.shape[0]

test_pix = test_imgs.reshape(n_test, -1)
test_z = batched_vmap(lambda im: flow.flatten_latent(ae.encode(im)),
                      jnp.expand_dims(test_imgs, axis=1))

if CALIB_POOL == "test":
    calib_pix, calib_z = test_pix, test_z
else:
    calib_imgs = np.asarray(dset_full.with_format("numpy")["sci_subtracted"], dtype=np.float32)
    calib_pix = calib_imgs.reshape(calib_imgs.shape[0], -1)
    calib_z = batched_vmap(lambda im: flow.flatten_latent(ae.encode(im)),
                           jnp.expand_dims(calib_imgs, axis=1))

n_calib = calib_pix.shape[0]
assert n_calib >= 2 * N_EVAL, f"pool of {n_calib} is too small for disjoint halves of {N_EVAL}"
print(f"Calibration pool {CALIB_POOL!r}: {n_calib} images, "
      f"~{N_EVAL ** 2 / n_calib:.0f}/{N_EVAL} images shared between two splits")
print(f"Test pool: {n_test} images, latent dim {test_z.shape[1]}, pixel dim {test_pix.shape[1]}")

# 4) generated pool for test 2, built once (decode + PSF convolution is costly).
# Generated image i uses the PSF of test galaxy i.
key, sk = jax.random.split(key)
gen_pool_z = flow.sample(key=sk, sample_shape=(n_test,))
gen_pool_pix = batched_vmap(
    lambda z, psf: ae.convolve(ae.decode(flow.unflatten_latent(z)), psf),
    gen_pool_z, jnp.expand_dims(test_psf, axis=1),
).reshape(n_test, -1)

# ae.convolve draws at the AE's (nx, ny): catch a mismatched checkpoint early
assert gen_pool_pix.shape[1] == test_pix.shape[1], (
    f"generated images are {gen_pool_pix.shape[1]}-D but real ones are {test_pix.shape[1]}-D: "
    f"check the autoencoder's nx/ny against the stamp size"
)


# --- draw functions: one (x, y) pair per split ---
# A and B come from one draw without replacement, so they are disjoint: shared images
# would push the p-values towards 1.

def draw_calib_image(rng):
    idx = rng.choice(n_calib, size=2 * N_EVAL, replace=False)
    return calib_pix[idx[:N_EVAL]], calib_pix[idx[N_EVAL:]]


def draw_calib_latent(rng):
    idx = rng.choice(n_calib, size=2 * N_EVAL, replace=False)
    return calib_z[idx[:N_EVAL]], calib_z[idx[N_EVAL:]]


def draw_flow_vs_latent(rng):
    # fresh flow samples at every split (cheap in latent space)
    global key
    key, sub = jax.random.split(key)
    z = np.asarray(flow.sample(key=sub, sample_shape=(N_EVAL,)))
    idx = rng.choice(n_test, size=N_EVAL, replace=False)
    return z, test_z[idx]


def draw_pipeline_vs_image(rng):
    # disjoint indices, so generated and real images never share a PSF
    idx = rng.choice(n_test, size=2 * N_EVAL, replace=False)
    return gen_pool_pix[idx[N_EVAL:]], test_pix[idx[:N_EVAL]]


# --- Test 0: calibration, real vs real (must be uniform for tests 1-2 to mean anything) ---
pvals_calib_img = pqm_split_pvalues(draw_calib_image, "test0 image ")
report_split_pvalues("Test 0 image", pvals_calib_img)
plot_split_uniformity(
    "test0_calibration_real-vs-real_image-space",
    "Test 0 - calibration (real vs real), image pixel space",
    f"x: real(A) raw pixels  |  y: real(B) raw pixels  (no model)"
    f"  |  pool={CALIB_POOL!r} ({n_calib} imgs)",
    pvals_calib_img,
)
plot_chi2(
    "test0_calibration_real-vs-real_image-space",
    "Test 0 - calibration (real vs real), image pixel space",
    f"x: real(A) raw pixels  |  y: real(B) raw pixels  (no model)"
    f"  |  pool={CALIB_POOL!r} ({n_calib} imgs)",
    pvals_calib_img,
)

pvals_calib_latent = pqm_split_pvalues(draw_calib_latent, "test0 latent")
report_split_pvalues("Test 0 latent", pvals_calib_latent)
plot_split_uniformity(
    "test0_calibration_real-vs-real_latent-space",
    "Test 0 - calibration (real vs real), flow latent space",
    f"x: real(A) -> AE.encode  |  y: real(B) -> AE.encode  (encoder only, no flow)"
    f"  |  pool={CALIB_POOL!r} ({n_calib} imgs)",
    pvals_calib_latent,
)
plot_chi2(
    "test0_calibration_real-vs-real_latent-space",
    "Test 0 - calibration (real vs real), flow latent space",
    f"x: real(A) -> AE.encode  |  y: real(B) -> AE.encode  (encoder only, no flow)"
    f"  |  pool={CALIB_POOL!r} ({n_calib} imgs)",
    pvals_calib_latent,
)

# --- Test 1: latent space, what the flow models directly ---
pvals_latent = pqm_split_pvalues(draw_flow_vs_latent, "test1 latent")
report_split_pvalues("Test 1 latent", pvals_latent)
plot_split_uniformity(
    "test1_flow-samples-vs-real_latent-space",
    "Test 1 - flow samples vs real, flow latent space",
    "x: flow.sample  (flow only)  |  y: real test-split -> AE.encode  (encoder only)",
    pvals_latent,
)
plot_chi2(
    "test1_flow-samples-vs-real_latent-space",
    "Test 1 - flow samples vs real, flow latent space",
    "x: flow.sample  (flow only)  |  y: real test-split -> AE.encode  (encoder only)",
    pvals_latent,
)

# --- Test 2: image space, full pipeline ---
pvals_img = pqm_split_pvalues(draw_pipeline_vs_image, "test2 image ")
report_split_pvalues("Test 2 image", pvals_img)
plot_split_uniformity(
    "test2_full-pipeline-gen-vs-real_image-space",
    "Test 2 - full pipeline vs real, image pixel space",
    "x: flow.sample -> AE.decode -> AE.convolve(PSF)  |  y: real test-split raw pixels",
    pvals_img,
)
plot_chi2(
    "test2_full-pipeline-gen-vs-real_image-space",
    "Test 2 - full pipeline vs real, image pixel space",
    "x: flow.sample -> AE.decode -> AE.convolve(PSF)  |  y: real test-split raw pixels",
    pvals_img,
)

print(f"\nFigures saved to {RESULTS_DIR}")
