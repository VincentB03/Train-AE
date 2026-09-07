#!/usr/bin/env python
"""Verifies that the latent flow generates within the density of the Hugging Face dataset,
using PQMass (x = generated samples, y = real test data)."""
from pathlib import Path

import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2, uniform

from datasets import load_dataset
from pqm import pqm_pvalue, pqm_chi2

from pshear.utils import load_galaxy_autoencoder, load_flow, fetch_wandb_checkpoint

# repo-root-relative, independent of $SCRATCH (unlike experiments.utils.PATH):
# both the checkpoint cache and the output figures stay next to the code.
ROOT = Path(".")

RESULTS_DIR = ROOT / "PQM_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# --- adapt to your run ---
WANDB_ENTITY = "vincentb03-imt-atlantique"
AE_RUN_PATH = f"{WANDB_ENTITY}/Test-AE-partial-3/i1pf186a"
AE_EPOCH = 2000
FLOW_RUN_PATH = f"{WANDB_ENTITY}/pshear-euclid-flow/95f2vnu6"
FLOW_EPOCH = 50
DATASET_NAME = "VincentB03/euclid-Q1-VF"
N_EVAL = 2000                              # number of samples for the test

# same convention as galaxy-morphometrics' WandBGalaxyAutoencoder/Flow:
# fetches+caches under ROOT/wandb_weights/<run_id>/epoch_<epoch>/, skipping
# the WandB API entirely if that directory is already pre-populated.
AE_MODEL_PATH = fetch_wandb_checkpoint(AE_RUN_PATH, AE_EPOCH, cache_dir=ROOT / "wandb_weights")
FLOW_MODEL_PATH = fetch_wandb_checkpoint(FLOW_RUN_PATH, FLOW_EPOCH, cache_dir=ROOT / "wandb_weights")

def plot_pqm_diagnostics(slug, title, pipeline, chi2_vals, pvals, dof):
    """Reproduces the diagnostic plots from the PQMass repo notebooks:
    chi2 histogram vs chi2(dof) pdf, and p-value histogram vs uniform pdf.

    ``slug`` is the descriptive file stem (-> ``PQM_results/<slug>_{chi2,pvalue}.png``),
    ``title`` the human-readable caption drawn on both figures, and ``pipeline`` the
    sub-caption spelling out which part of the pipeline produces each PQMass sample
    (x = generated, y = reference), e.g. "x: flow.sample -> AE.decode -> AE.convolve(PSF)"."""
    for values, ref_pdf, xlabel, kind in (
        (chi2_vals, lambda g: chi2.pdf(g, df=dof), r"$\chi^2_{\rm PQM}$", "chi2"),
        (pvals, lambda g: uniform.pdf(g), "p-value", "pvalue"),
    ):
        fig, ax = plt.subplots(figsize=(6, 3.8))
        if kind == "chi2":
            ax.hist(values, bins=20, density=True)
            grid = np.linspace(min(values), max(values), 200)
        else:
            ax.hist(values, bins=10, density=True, range=(0, 1))
            grid = np.linspace(0, 1, 100)
        ax.plot(grid, ref_pdf(grid), color="red")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Frequency")
        ax.set_title(f"{title}\n{pipeline}", fontsize=9)
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / f"{slug}_{kind}.png", dpi=150)
        plt.close(fig)


key = jax.random.key(0)

# 1) frozen models
ae = load_galaxy_autoencoder(AE_MODEL_PATH, epoch=AE_EPOCH)
ae = eqx.nn.inference_mode(ae, value=True)

flow = load_flow(FLOW_MODEL_PATH, epoch=FLOW_EPOCH)
flow = eqx.nn.inference_mode(flow, value=True)

# 2) real data - SAME test split as train_flow.py (seed=42) to avoid
# comparing against data already seen by the flow
dset = load_dataset(DATASET_NAME, split="train", keep_in_memory=True)
dset = dset.train_test_split(test_size=0.1, seed=42)
dset_test = dset["test"].with_format("numpy")

# 3) generation -> z_gen, then decode. PSF convolution is deferred to section 5:
# it uses partition B's PSFs so that test 2's generated sample never shares PSF
# realisations with its real reference (partition A), which would otherwise erase
# PSF-driven scatter from the comparison and bias the test toward passing.
key, sk = jax.random.split(key)
z_gen = flow.sample(key=sk, sample_shape=(N_EVAL,))
gen_imgs_preconv = jax.vmap(ae.decode)(flow.unflatten_latent(z_gen))

z_gen_np = np.asarray(z_gen)

# 4) real reference data: two disjoint halves A/B of the test split, drawn
# uniformly without replacement. A is the real reference for EVERY test; B is a
# second independent real sample. Test 0 (A vs B) sanity-checks the PQMass setup
# itself (num_refs, re_tessellation, ...) independently of the model and should
# recover p-value ~ 0.5 and chi2/DoF ~ 1, since both halves share one distribution.
idx_calib = np.random.default_rng(1).choice(len(dset_test), size=2 * N_EVAL, replace=False)
idx_calib_a, idx_calib_b = idx_calib[:N_EVAL], idx_calib[N_EVAL:]
calib_imgs_a = jnp.expand_dims(dset_test[idx_calib_a]["sci_subtracted"], axis=1)
calib_imgs_b = jnp.expand_dims(dset_test[idx_calib_b]["sci_subtracted"], axis=1)

z_calib_a = flow.flatten_latent(jax.vmap(ae.encode)(calib_imgs_a))
z_calib_b = flow.flatten_latent(jax.vmap(ae.encode)(calib_imgs_b))
z_calib_a_np, z_calib_b_np = np.asarray(z_calib_a), np.asarray(z_calib_b)
calib_imgs_a_np = np.asarray(calib_imgs_a).reshape(N_EVAL, -1)
calib_imgs_b_np = np.asarray(calib_imgs_b).reshape(N_EVAL, -1)

# 5) generated images for test 2: convolve with partition B's PSFs, then compare
# against partition A. x (generated) and y (real, = A) rest on DISJOINT real subsets
# with independently drawn PSFs, so test 2 shares its null with test 0 image (A vs B).
psf_b = jnp.expand_dims(dset_test[idx_calib_b]["psf_residual"], axis=1)
gen_imgs = jax.vmap(ae.convolve)(gen_imgs_preconv, psf_b)
gen_imgs_np = np.asarray(gen_imgs).reshape(N_EVAL, -1)

# --- Test 0: calibration (real vs real) ---
pvals_calib_latent = pqm_pvalue(z_calib_a_np, z_calib_b_np, num_refs=100, re_tessellation=1000, z_score_norm=True)
chi2_calib_latent = pqm_chi2(z_calib_a_np, z_calib_b_np, num_refs=100, re_tessellation=1000, z_score_norm=True)
print("Calib latent -> p-value mean/std:", np.mean(pvals_calib_latent), np.std(pvals_calib_latent))
print("Calib latent -> chi2/DoF mean:", np.mean(chi2_calib_latent) / 99)
plot_pqm_diagnostics(
    "test0_calibration_real-vs-real_latent-space",
    "Test 0 - calibration (real vs real), flow latent space (16-D)",
    "x: real(A) -> AE.encode -> flatten_latent  |  y: real(B) -> AE.encode -> flatten_latent  (no flow)",
    chi2_calib_latent, pvals_calib_latent, dof=99,
)

pvals_calib_img = pqm_pvalue(calib_imgs_a_np, calib_imgs_b_np, num_refs=100, re_tessellation=1000, z_score_norm=True)
chi2_calib_img = pqm_chi2(calib_imgs_a_np, calib_imgs_b_np, num_refs=100, re_tessellation=1000, z_score_norm=True)
print("Calib image  -> p-value mean/std:", np.mean(pvals_calib_img), np.std(pvals_calib_img))
print("Calib image  -> chi2/DoF mean:", np.mean(chi2_calib_img) / 99)
plot_pqm_diagnostics(
    "test0_calibration_real-vs-real_image-space",
    "Test 0 - calibration (real vs real), image pixel space",
    "x: real(A) raw pixels  |  y: real(B) raw pixels  (no encoder, no flow, no decoder)",
    chi2_calib_img, pvals_calib_img, dof=99,
)

# --- Test 1: latent space (what the flow models directly), generated vs real partition A ---
pvals_latent = pqm_pvalue(z_gen_np, z_calib_a_np, num_refs=100, re_tessellation=1000, z_score_norm=True)
chi2_latent = pqm_chi2(z_gen_np, z_calib_a_np, num_refs=100, re_tessellation=1000, z_score_norm=True)
print("Latent  -> p-value mean/std:", np.mean(pvals_latent), np.std(pvals_latent))
print("Latent  -> chi2/DoF mean:", np.mean(chi2_latent) / 99)
plot_pqm_diagnostics(
    "test1_flow-samples-vs-real_latent-space",
    "Test 1 - flow samples vs real (partition A), flow latent space (16-D)",
    "x: flow.sample  (flow only)  |  y: real partition-A -> AE.encode -> flatten_latent  (encoder only)",
    chi2_latent, pvals_latent, dof=99,
)

# --- Test 2: image space (full pipeline), generated vs real partition A ---
pvals_img = pqm_pvalue(gen_imgs_np, calib_imgs_a_np, num_refs=100, re_tessellation=1000, z_score_norm=True)
chi2_img = pqm_chi2(gen_imgs_np, calib_imgs_a_np, num_refs=100, re_tessellation=1000, z_score_norm=True)
print("Image   -> p-value mean/std:", np.mean(pvals_img), np.std(pvals_img))
print("Image   -> chi2/DoF mean:", np.mean(chi2_img) / 99)
plot_pqm_diagnostics(
    "test2_full-pipeline-gen-vs-real_image-space",
    "Test 2 - full pipeline (flow + decode + PSF convolve) vs real (partition A), image pixel space",
    "x: flow.sample -> AE.decode -> AE.convolve(partition-B PSF)  (flow + decoder)  |  y: real partition-A raw pixels",
    chi2_img, pvals_img, dof=99,
)

print(f"Figures saved to {RESULTS_DIR}")
