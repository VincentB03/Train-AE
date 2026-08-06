#!/usr/bin/env python
"""Verifies that the latent flow generates within the density of the Hugging Face dataset,
using PQMass (x = generated samples, y = real test data)."""
import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np

from datasets import load_dataset
from pqm import pqm_pvalue, pqm_chi2

from pshear.utils import load_galaxy_autoencoder, load_flow
from experiments.utils import PATH

# --- adapt to your run ---
AE_MODEL_PATH = PATH / "wandb_weights" / "i1pf186a" / "epoch_2000"
AE_EPOCH = 2000
FLOW_MODEL_PATH = PATH / "wandb_weights" / "95f2vnu6" / "epoch_50"
FLOW_EPOCH = 50
DATASET_NAME = "VincentB03/euclid-Q1-V2"
N_EVAL = 2000                              # number of samples for the test

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

# --- Test 1: latent space (what the flow models directly) ---
pvals_latent = pqm_pvalue(z_gen_np, z_real_np, num_refs=100, re_tessellation=1000)
chi2_latent = pqm_chi2(z_gen_np, z_real_np, num_refs=100, re_tessellation=1000)
print("Latent  -> p-value mean/std:", np.mean(pvals_latent), np.std(pvals_latent))
print("Latent  -> chi2/DoF mean:", np.mean(chi2_latent) / 99)

# --- Test 2: image space (full pipeline) ---
pvals_img = pqm_pvalue(gen_imgs_np, real_imgs_np, num_refs=100, re_tessellation=1000)
chi2_img = pqm_chi2(gen_imgs_np, real_imgs_np, num_refs=100, re_tessellation=1000)
print("Image   -> p-value mean/std:", np.mean(pvals_img), np.std(pvals_img))
print("Image   -> chi2/DoF mean:", np.mean(chi2_img) / 99)
