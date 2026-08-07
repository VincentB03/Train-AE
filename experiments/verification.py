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
DATASET_NAME = "VincentB03/euclid-Q1-V2"
N_EVAL = 2000                              # number of samples for the test

# same convention as galaxy-morphometrics' WandBGalaxyAutoencoder/Flow:
# fetches+caches under ROOT/wandb_weights/<run_id>/epoch_<epoch>/, skipping
# the WandB API entirely if that directory is already pre-populated.
AE_MODEL_PATH = fetch_wandb_checkpoint(AE_RUN_PATH, AE_EPOCH, cache_dir=ROOT / "wandb_weights")
FLOW_MODEL_PATH = fetch_wandb_checkpoint(FLOW_RUN_PATH, FLOW_EPOCH, cache_dir=ROOT / "wandb_weights")

def plot_pqm_diagnostics(name, chi2_vals, pvals, dof):
    """Reproduces the diagnostic plots from the PQMass repo notebooks:
    chi2 histogram vs chi2(dof) pdf, and p-value histogram vs uniform pdf."""
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(chi2_vals, bins=20, density=True)
    x = np.linspace(min(chi2_vals), max(chi2_vals), 200)
    ax.plot(x, chi2.pdf(x, df=dof), color="red")
    ax.set_xlabel(r"$\chi^2_{\rm PQM}$")
    ax.set_ylabel("Frequency")
    ax.set_title(name)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"{name}_chi2.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(pvals, bins=10, density=True, range=(0, 1))
    x = np.linspace(0, 1, 100)
    ax.plot(x, uniform.pdf(x), color="red")
    ax.set_xlabel("p-value")
    ax.set_ylabel("Frequency")
    ax.set_title(name)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"{name}_pvalue.png", dpi=150)
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

idx = np.random.default_rng(0).choice(len(dset_test), size=N_EVAL, replace=False)
real_imgs = jnp.expand_dims(dset_test[idx]["sci_subtracted"], axis=1)  # (N,1,H,W)
real_psf = jnp.expand_dims(dset_test[idx]["psf_residual"], axis=1)

# 3) real encoding -> z_real (flow latent space)
z_real = jax.vmap(ae.encode)(real_imgs)
z_real = flow.flatten_latent(z_real)          # (N, 16)

# 4) generation -> z_gen, then decode+convolve -> generated images
key, sk = jax.random.split(key)
z_gen = flow.sample(key=sk, sample_shape=(N_EVAL,))
gen_imgs = jax.vmap(ae.decode)(flow.unflatten_latent(z_gen))
gen_imgs = jax.vmap(ae.convolve)(gen_imgs, real_psf)  # reuses real PSFs

z_real_np, z_gen_np = np.asarray(z_real), np.asarray(z_gen)
real_imgs_np = np.asarray(real_imgs).reshape(N_EVAL, -1)
gen_imgs_np = np.asarray(gen_imgs).reshape(N_EVAL, -1)

# 5) calibration data: two disjoint real halves, to sanity-check the PQMass
# setup itself (num_refs, re_tessellation, ...) independently of the model.
# A well-calibrated test should recover p-value ~ 0.5 and chi2/DoF ~ 1 here,
# since both halves are drawn from the same real distribution.
idx_calib = np.random.default_rng(1).choice(len(dset_test), size=2 * N_EVAL, replace=False)
idx_calib_a, idx_calib_b = idx_calib[:N_EVAL], idx_calib[N_EVAL:]
calib_imgs_a = jnp.expand_dims(dset_test[idx_calib_a]["sci_subtracted"], axis=1)
calib_imgs_b = jnp.expand_dims(dset_test[idx_calib_b]["sci_subtracted"], axis=1)

z_calib_a = flow.flatten_latent(jax.vmap(ae.encode)(calib_imgs_a))
z_calib_b = flow.flatten_latent(jax.vmap(ae.encode)(calib_imgs_b))
z_calib_a_np, z_calib_b_np = np.asarray(z_calib_a), np.asarray(z_calib_b)
calib_imgs_a_np = np.asarray(calib_imgs_a).reshape(N_EVAL, -1)
calib_imgs_b_np = np.asarray(calib_imgs_b).reshape(N_EVAL, -1)

# --- Test 0: calibration (real vs real) ---
pvals_calib_latent = pqm_pvalue(z_calib_a_np, z_calib_b_np, num_refs=100, re_tessellation=1000)
chi2_calib_latent = pqm_chi2(z_calib_a_np, z_calib_b_np, num_refs=100, re_tessellation=1000)
print("Calib latent -> p-value mean/std:", np.mean(pvals_calib_latent), np.std(pvals_calib_latent))
print("Calib latent -> chi2/DoF mean:", np.mean(chi2_calib_latent) / 99)
plot_pqm_diagnostics("calib_latent", chi2_calib_latent, pvals_calib_latent, dof=99)

pvals_calib_img = pqm_pvalue(calib_imgs_a_np, calib_imgs_b_np, num_refs=100, re_tessellation=1000)
chi2_calib_img = pqm_chi2(calib_imgs_a_np, calib_imgs_b_np, num_refs=100, re_tessellation=1000)
print("Calib image  -> p-value mean/std:", np.mean(pvals_calib_img), np.std(pvals_calib_img))
print("Calib image  -> chi2/DoF mean:", np.mean(chi2_calib_img) / 99)
plot_pqm_diagnostics("calib_image", chi2_calib_img, pvals_calib_img, dof=99)

# --- Test 1: latent space (what the flow models directly) ---
pvals_latent = pqm_pvalue(z_gen_np, z_real_np, num_refs=100, re_tessellation=1000)
chi2_latent = pqm_chi2(z_gen_np, z_real_np, num_refs=100, re_tessellation=1000)
print("Latent  -> p-value mean/std:", np.mean(pvals_latent), np.std(pvals_latent))
print("Latent  -> chi2/DoF mean:", np.mean(chi2_latent) / 99)
plot_pqm_diagnostics("latent", chi2_latent, pvals_latent, dof=99)

# --- Test 2: image space (full pipeline) ---
pvals_img = pqm_pvalue(gen_imgs_np, real_imgs_np, num_refs=100, re_tessellation=1000)
chi2_img = pqm_chi2(gen_imgs_np, real_imgs_np, num_refs=100, re_tessellation=1000)
print("Image   -> p-value mean/std:", np.mean(pvals_img), np.std(pvals_img))
print("Image   -> chi2/DoF mean:", np.mean(chi2_img) / 99)
plot_pqm_diagnostics("image", chi2_img, pvals_img, dof=99)

print(f"Figures saved to {RESULTS_DIR}")
