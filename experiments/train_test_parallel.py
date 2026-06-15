#!/usr/bin/env python

import os
import jax
import optax
import equinox as eqx
import numpy as np
import matplotlib.pyplot as plt

# JAX Sharding and Parallelization imports
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils

# PyTorch DataLoader for optimized pipeline
import torch
from torch.utils.data import DataLoader

from pshear.galaxy import GalaxyAutoEncoderLoss, make_galaxy_autoencoder
from pshear.utils import dump_galaxy_autoencoder
from datasets import load_dataset
from experiments.utils import PATH, plot_ae_residuals

import wandb
from tqdm import tqdm

CONFIG = {
    "use_jax_galsim": True,
    "minimum_fft_size": 64, 
    "nx": 64,
    "ny": 64,
    "scale": 0.03,
    "in_channels": 1,
    "latent_channels": 1,
    "hid_channels": (32, 32, 64, 128, 256), 
    "hid_blocks": (2, 2, 2, 2, 2),          
    "attention_heads": {5: 4},              
    "patch_size": 1,
    "stride": 2,
    "dropout": 0.05,
    "kernel_size": 3,
    "batch_size": 128,    
    "epochs": 500,        
    "learning_rate": 1e-5,
    "losses": ["likelihood", "tv"],
    "weights": [1.0, 0.01],
    "log_freq": 50,
    "num_workers": 4, # Adding num_workers for DataLoader
}

def ema_update(params, ema_params, decay):
    return jax.tree_util.tree_map(
        lambda p, e: decay * e + (1.0 - decay) * p, params, ema_params
    )

def numpy_collate(batch):
    """
    Collate function to prepare the batch in parallel on CPU workers.
    Replaces the manual np.expand_dims and normalization in the training loop.
    """
    sci_subtracted = np.array([item["sci_subtracted"] for item in batch])
    psf_stamp = np.array([item["psf_stamp"] for item in batch])
    noise_map = np.array([item["noise_map"] for item in batch])
    binary_mask = np.array([item["binary_mask"] for item in batch])

    img = np.expand_dims(sci_subtracted, axis=1)
    psf = np.expand_dims(psf_stamp, axis=1)
    rms = np.expand_dims(noise_map, axis=1)
    mask = np.expand_dims(binary_mask, axis=1)

    norm_factor = np.max(np.abs(img), axis=(1, 2, 3), keepdims=True)
    norm_factor = np.where(norm_factor == 0, 1.0, norm_factor)
    
    img = img / norm_factor
    rms = rms / norm_factor

    return {
        "sci_subtracted": img,
        "psf_stamp": psf,
        "rms": rms,
        "mask": mask
    }

def train(runid: str):
    # Initialize distributed JAX if available (SLURM, multi-node)
    try:
        jax.distributed.initialize()
    except Exception as e:
        print(f"Distributed init failed or already initialized: {e}. Proceeding locally.")

    num_devices = jax.device_count()
    print(f"Running on {num_devices} JAX devices.")

    # Create a Mesh for Data Parallelism
    devices = mesh_utils.create_device_mesh((num_devices,))
    mesh = Mesh(devices, axis_names=('batch',))
    
    # We shard data along the 'batch' dimension, keeping other dimensions replicated
    data_sharding = NamedSharding(mesh, P('batch'))
    # Parameters and states are replicated across all devices
    replicated_sharding = NamedSharding(mesh, P())

    run = wandb.init(
        project="Generative-Euclid",
        name="ae-test-Q1-JZ_1-parallel",
        id=runid,
        resume="allow",
        dir=PATH,
        config=CONFIG,
    )

    exp_path = PATH / f"runs/{run.name}_{run.id}"
    exp_path.mkdir(parents=True, exist_ok=True)
    cfg = run.config
    
    print("Loading Dataset from Hugging Face")
    dset = load_dataset("VincentB03/euclid-Q1-V2", split="train", keep_in_memory=True)
    
    dset = dset.train_test_split(test_size=0.1, seed=42)
    dset_train = dset["train"]
    dset_test = dset["test"]

    # Ensure batch size is divisible by number of devices
    assert cfg.batch_size % num_devices == 0, "Batch size must be divisible by the number of devices."

    # Using PyTorch DataLoaders for multi-processing data loading
    train_loader = DataLoader(
        dset_train,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=numpy_collate,
        drop_last=True
    )
    
    test_loader = DataLoader(
        dset_test,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=numpy_collate,
        drop_last=True
    )

    key = jax.random.PRNGKey(0)

    model = make_galaxy_autoencoder(
        use_jax_galsim=cfg.use_jax_galsim, minimum_fft_size=cfg.minimum_fft_size,
        nx=cfg.nx, ny=cfg.ny, scale=cfg.scale, in_channels=cfg.in_channels,
        latent_channels=cfg.latent_channels, hid_channels=cfg.hid_channels,
        hid_blocks=cfg.hid_blocks, attention_heads=cfg.attention_heads,
        patch_size=cfg.patch_size, stride=cfg.stride, dropout=cfg.dropout,
        kernel_size=cfg.kernel_size, key=key,
    )

    params, static = eqx.partition(model, eqx.is_array)
    ema_params = params

    # Distribute initial parameters
    params = jax.device_put(params, replicated_sharding)
    ema_params = jax.device_put(ema_params, replicated_sharding)

    loss_fn = jax.vmap(
        GalaxyAutoEncoderLoss(losses=cfg.losses, weights=cfg.weights),
        in_axes=(None, 0, 0, 0, 0, None, None),
    )

    @eqx.filter_jit
    def loss(params, batch, key, activate):
        model = eqx.combine(params, static)
        return loss_fn(
            model, 
            batch["sci_subtracted"], 
            batch["psf_stamp"], 
            batch["rms"], 
            batch["mask"],
            key, activate
        ).mean()

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=cfg.learning_rate, b1=0.9, b2=0.95, weight_decay=1e-4),
    )
    opt_state = optimizer.init(params)
    opt_state = jax.device_put(opt_state, replicated_sharding)

    @eqx.filter_jit
    def opt_step(params, ema_params, opt_state, batch, key, activate=0.0):
        loss_value, grads = jax.value_and_grad(loss)(params, batch, key, activate)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        ema_params = ema_update(params, ema_params, decay=0.999)
        return loss_value, params, ema_params, opt_state

    activate = 0.0
    for epoch in range(cfg.epochs):
        print(f"Epoch {epoch+1}/{cfg.epochs}")

        if epoch == 100: 
            activate = 1.0
            
        train_losses = []
        for batch in tqdm(train_loader, desc="Training"):
            # Put data on devices according to sharding spec (Data Parallelism)
            clean_batch = jax.tree_util.tree_map(lambda x: jax.device_put(x, data_sharding), batch)

            key, subkey = jax.random.split(key, 2)
            
            loss_value, params, ema_params, opt_state = opt_step(
                params, ema_params, opt_state, clean_batch, subkey, activate=activate
            )
            train_losses.append(loss_value)

        # compute mean train loss. block until ready to avoid async JAX hiding memory
        loss_train = np.mean(jax.device_get(train_losses)) if train_losses else 0.0

        test_losses = []
        for batch in tqdm(test_loader, desc="Testing"):
            clean_batch = jax.tree_util.tree_map(lambda x: jax.device_put(x, data_sharding), batch)
            
            subkey, key = jax.random.split(key, 2)
            loss_value = loss(params, clean_batch, subkey, 0.0)
            test_losses.append(loss_value)

        loss_test = np.mean(jax.device_get(test_losses)) if test_losses else 0.0

        if (epoch + 1) % cfg.log_freq == 0:
            model = eqx.combine(params, static)
            model = eqx.nn.inference_mode(model, True)
            
            # Fetch a sample batch for plotting (putting it on single device for plot logic)
            plot_batch = jax.tree_util.tree_map(lambda x: jax.device_get(x)[0:1], batch)
            
            y, _, _ = jax.vmap(model)(plot_batch["sci_subtracted"], plot_batch["psf_stamp"])
            
            x = plot_ae_residuals(plot_batch, y)

            plot_path = exp_path / f"residuals_epoch_{epoch+1}.png"

            run.log({
                "loss_train": loss_train,
                "loss_test": loss_test,
                "fit_and_residuals": wandb.Image(x),
            })

            dump_galaxy_autoencoder(exp_path, model, epoch + 1, CONFIG)

            wandb.save(str(exp_path / "*"), base_path=str(exp_path.parent))
        else:
            run.log({"loss_train": loss_train, "loss_test": loss_test})
            
    artifact = wandb.Artifact(
        name=f"galaxy-ae-{run.id}", 
        type="model",
        metadata=CONFIG
    )
    artifact.add_dir(str(exp_path))
    run.log_artifact(artifact)

    wandb.finish()

if __name__ == "__main__":
    runid = wandb.util.generate_id()
    train(runid=runid)
