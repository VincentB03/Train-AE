#!/usr/bin/env python
"""Checks with PQMass whether the latent flow generates within the density of the
Hugging Face dataset, at four points along the pipeline:

    Test 0 image   real vs real, raw pixels          -- calibration, no model involved
    Test 0 latent  real vs real, AE latents          -- calibration, encoder involved
    Test 1         flow samples vs encoded real      -- what the flow models directly
    Test 2         full pipeline vs real, raw pixels -- flow + decoder + PSF

Every test draws N_SPLITS fresh (x, y) pairs and keeps ONE p-value per pair; under H0
that collection is uniform on [0, 1]. The two calibration tests set the reference tests
1-2 are read against.

Why not pqm_pvalue(..., re_tessellation=k): that call holds x and y fixed and only
re-draws the Voronoi tessellation, so its k p-values share one sample-level fluctuation
and count as a single independent observation. Measured at N=2000, d=4096, num_refs=100
under H0, the spread of the per-pair mean is 0.035 vs a within-pair spread of 0.27 --
tessellation noise dominates, so that histogram looks uniform regardless of the pair and
certifies nothing. Uniformity is a property of the variation ACROSS pairs, which is what
N_SPLITS samples.
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

# repo-root-relative, independent of $SCRATCH (unlike experiments.utils.PATH):
# both the checkpoint cache and the output figures stay next to the code.
ROOT = Path(".")

RESULTS_DIR = ROOT / "PQM_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# --- adapt to your run ---
WANDB_ENTITY = "vincentb03-imt-atlantique"
AE_RUN_PATH = f"{WANDB_ENTITY}/AE-partial-droppedDB/i344nq38"
AE_EPOCH = 2000
FLOW_RUN_PATH = f"{WANDB_ENTITY}/pshear-euclid-flow-dropped-db/9i28jqsm"
FLOW_EPOCH = 500
DATASET_NAME = "VincentB03/euclid-Q1-VF"

N_EVAL = 2000        # samples per side of one PQMass comparison
N_SPLITS = 200       # (x, y) pairs per test; one p-value each
NUM_REFS = 100
DOF = NUM_REFS - 1   # pqm rescales its chi2 to always use this dof
SPLIT_SEED = 20250909
ENCODE_BATCH = 500   # batch size for the one-off encode/decode/convolve passes

# Pool the two calibration tests draw their (A, B) partitions from:
#   "test" -> the 10% test split (~5020 images), same pool as tests 1-2.
#   "all"  -> the full 50203-image dataset: overlap between two splits drops to
#             ~N_EVAL**2/50203 ~ 80/2000 (vs ~800/2000 for "test"), so splits are
#             near-independent. Costs ~0.8 GB RAM for the flattened pixel pool.
#             Caveat: the latent half runs through ae.encode, so train-split images
#             then get encoded by an AE trained on them. Null still holds (both sides
#             identically distributed), but it's no longer test 1's exact reference.
CALIB_POOL = "all"

# pqm draws its references through the legacy global numpy RNG, which is what makes the
# tessellations reproducible. Partitions use their own Generator, seeded from SPLIT_SEED.
np.random.seed(0)

# same convention as galaxy-morphometrics' WandBGalaxyAutoencoder/Flow:
# fetches+caches under ROOT/wandb_weights/<run_id>/epoch_<epoch>/, skipping
# the WandB API entirely if that directory is already pre-populated.
AE_MODEL_PATH = fetch_wandb_checkpoint(AE_RUN_PATH, AE_EPOCH, cache_dir=ROOT / "wandb_weights")
FLOW_MODEL_PATH = fetch_wandb_checkpoint(FLOW_RUN_PATH, FLOW_EPOCH, cache_dir=ROOT / "wandb_weights")


def batched_vmap(fn, *arrays, batch=ENCODE_BATCH):
    """vmap ``fn`` over the leading axis in chunks, so the one-off passes over the whole
    pool (encode, decode, PSF convolution) need not fit in device memory at once."""
    vfn = jax.vmap(fn)
    n = arrays[0].shape[0]
    out = [np.asarray(vfn(*(a[i:i + batch] for a in arrays))) for i in range(0, n, batch)]
    return np.concatenate(out, axis=0)


def pqm_split_pvalues(draw_pair, desc, n_splits=N_SPLITS, num_refs=NUM_REFS):
    """Draw ``n_splits`` fresh (x, y) pairs and return one p-value per pair.

    ``draw_pair(rng)`` returns the (x, y) arrays of one split, both (N, D). Every test
    reseeds from SPLIT_SEED, so their p-values are paired split by split.

    Only pqm_pvalue is called: pqm_chi2 returns chi2.isf(p_value, num_refs - 1), a
    deterministic function of the p-value, so calling it too would double the cost (the
    dominant term is two cdist calls, ~1 s at N=2000, d=4096, num_refs=100) and, since it
    re-draws its own tessellation, return chi2 values not paired with these p-values.
    Derive them with chi2.isf(pvals, DOF) instead.
    """
    rng = np.random.default_rng(SPLIT_SEED)
    pvals = np.empty(n_splits)
    for i in tqdm(range(n_splits), desc=desc):
        x, y = draw_pair(rng)
        pvals[i] = pqm_pvalue(x, y, num_refs=num_refs, z_score_norm=True)
    return pvals


def report_split_pvalues(label, pvals):
    """One console line per test. Under H0: mean 0.5, frac(p<0.05) 0.05, KS not rejecting,
    median chi2/DoF ~ 1.

    KS p-value is anti-conservative when the pool is too small for splits to be disjoint
    (overlapping pairs correlate their p-values) -- read it as an effect size, not a rule.
    """
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
    """The PQMass-repo figure: histogram of the PQM chi2 against the chi2(DOF) pdf.

    Derived from the same p-values via chi2.isf (what pqm_chi2 computes), so this is a
    reparametrisation of plot_split_uniformity's histogram, not an independent check --
    kept as the familiar view, and the tail reads better on this scale. p == 0 maps to
    +inf; dropped and counted in the title, along with points outside the x-range (which
    spans both the chi2(DOF) bulk and the data, so a strong rejection stays readable
    instead of flattening the reference pdf to zero).
    """
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
    """Left: binned counts vs expected, with +/-1 and +/-2 sigma binomial bands (makes
    ~20 counts/bin judgeable by eye). Right: empirical CDF vs diagonal with the KS 95%
    band (+/-1.36/sqrt(n)) -- a uniformity defect shows as a sustained excursion rather
    than one noisy bar. ``pipeline`` names which part of the pipeline made each sample.
    """
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

# 2) real data, SAME test split as train_flow.py (seed=42): the flow must never be
# compared against latents it was trained on.
dset_full = load_dataset(DATASET_NAME, split="train", keep_in_memory=True)
dset = dset_full.train_test_split(test_size=0.1, seed=42)
dset_test = dset["test"].with_format("numpy")

# 3) pools, materialised once so the per-split cost is a single tessellation.
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

# 4) generated pool for test 2, built once: generating fresh images per split would add
# a decode + jax_galsim convolution pass and dominate the ~1 s tessellation. Generated
# image i carries the PSF of test-pool galaxy i, so keeping the generated/reference
# index sets disjoint below also keeps their PSF realisations disjoint.
key, sk = jax.random.split(key)
gen_pool_z = flow.sample(key=sk, sample_shape=(n_test,))
gen_pool_pix = batched_vmap(
    lambda z, psf: ae.convolve(ae.decode(flow.unflatten_latent(z)), psf),
    gen_pool_z, jnp.expand_dims(test_psf, axis=1),
).reshape(n_test, -1)

# ae.convolve draws at the AE's own (nx, ny) (64x64 in the training config); catch a
# mismatched checkpoint here rather than deep inside pqm.
assert gen_pool_pix.shape[1] == test_pix.shape[1], (
    f"generated images are {gen_pool_pix.shape[1]}-D but real ones are {test_pix.shape[1]}-D: "
    f"check the autoencoder's nx/ny against the stamp size"
)


# --- draw functions: one (x, y) pair per split ---
# A and B always come from ONE draw without replacement, then halved, so they stay
# disjoint. Two independent draws would let them share ~N_EVAL**2/pool images; shared
# points fall in the same Voronoi cell, deflating chi2 and pushing p -> 1 -- a failure
# that looks exactly like "the two samples agree beautifully".

def draw_calib_image(rng):
    idx = rng.choice(n_calib, size=2 * N_EVAL, replace=False)
    return calib_pix[idx[:N_EVAL]], calib_pix[idx[N_EVAL:]]


def draw_calib_latent(rng):
    idx = rng.choice(n_calib, size=2 * N_EVAL, replace=False)
    return calib_z[idx[:N_EVAL]], calib_z[idx[N_EVAL:]]


def draw_flow_vs_latent(rng):
    # x is sampled fresh from the flow every split -- cheap at this latent size.
    global key
    key, sub = jax.random.split(key)
    z = np.asarray(flow.sample(key=sub, sample_shape=(N_EVAL,)))
    idx = rng.choice(n_test, size=N_EVAL, replace=False)
    return z, test_z[idx]


def draw_pipeline_vs_image(rng):
    # A = real reference, B = generated indices (hence their PSFs). Disjoint, so x and y
    # never share a PSF realisation -- sharing one would bias the test toward passing.
    idx = rng.choice(n_test, size=2 * N_EVAL, replace=False)
    return gen_pool_pix[idx[N_EVAL:]], test_pix[idx[:N_EVAL]]


# --- Test 0: calibration, real vs real. Must come out uniform for tests 1-2 to mean
# anything. ---
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
