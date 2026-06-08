#!/usr/bin/env python

import os
import jax
import optax
import equinox as eqx
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from pshear.galaxy import GalaxyAutoEncoderLoss, make_galaxy_autoencoder
from pshear.utils import dump_galaxy_autoencoder

from datasets import load_dataset
# no data augmentation 
from experiments.utils import PATH, plot_ae_residuals, plot_residual_histogram

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
    "epochs": 1200,        
    "learning_rate": 5e-5,
    "losses": ["likelihood", "tv"],
    "weights": [1.0, 1e-5],
    "log_freq": 100,
}


def ema_update(params, ema_params, decay):
    return jax.tree_util.tree_map(
        lambda p, e: decay * e + (1.0 - decay) * p, params, ema_params
    )

def train(runid: str):
    run = wandb.init(
        project="Generative-Euclid",
        name="ae-test-Q1-JZ_1",
        id=runid,
        resume="allow",
        dir=PATH,
        config=CONFIG,
    )

    exp_path = PATH / f"runs/{run.name}_{run.id}"
    exp_path.mkdir(parents=True, exist_ok=True)
    cfg = run.config
    
    print("Loading Dataset from Hugging Face")
    dset = load_dataset("VincentB03/euclid-Q1-V2", split="train", keep_in_memory=True) #Try keeping in memory for faster training
    
    dset = dset.train_test_split(test_size=0.1, seed=42)
    
    dset_train = dset["train"]
    dset_test = dset["test"]

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

    loss_fn = jax.vmap(
        GalaxyAutoEncoderLoss(losses=cfg.losses, weights=cfg.weights),
        in_axes=(None, 0, 0, 0, 0, None, None),
    )

    @jax.jit
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

    @jax.jit
    def opt_step(params, ema_params, opt_state, batch, key, activate=0.0):
        loss_value, grads = jax.value_and_grad(loss)(params, batch, key, activate)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        ema_params = ema_update(params, ema_params, decay=0.999)
        return loss_value, params, ema_params, opt_state
    
    ram_data = dset_train.with_format("numpy")[:]
    num_samples = len(ram_data["sci_subtracted"])

    X_sci_train = jnp.expand_dims(jnp.array(ram_data["sci_subtracted"], dtype=jnp.float32), axis=1)
    X_psf_train = jnp.expand_dims(jnp.array(ram_data["psf_stamp"], dtype=jnp.float32), axis=1)
    X_rms_train = jnp.expand_dims(jnp.array(ram_data["noise_map"], dtype=jnp.float32), axis=1)
    X_mask_train = jnp.expand_dims(jnp.array(ram_data["binary_mask"], dtype=jnp.float32), axis=1)

    ram_test = dset_test.with_format("numpy")[:]
    num_test_samples = len(ram_test["sci_subtracted"])

    X_sci_test = jnp.expand_dims(jnp.array(ram_test["sci_subtracted"], dtype=jnp.float32), axis=1)
    X_psf_test = jnp.expand_dims(jnp.array(ram_test["psf_stamp"], dtype=jnp.float32), axis=1)
    X_rms_test = jnp.expand_dims(jnp.array(ram_test["noise_map"], dtype=jnp.float32), axis=1)
    X_mask_test = jnp.expand_dims(jnp.array(ram_test["binary_mask"], dtype=jnp.float32), axis=1)

    activate = 0.0
    for epoch in tqdm(range(cfg.epochs)):

        if epoch == 100: 
            activate = 1.0
            
        losses = []

        indices = np.random.permutation(num_samples)

        for i in tqdm(range(0, num_samples - cfg.batch_size + 1, cfg.batch_size)):
            batch_idx = indices[i : i + cfg.batch_size]

            clean_batch = {
                "sci_subtracted": X_sci_train[batch_idx],
                "psf_stamp": X_psf_train[batch_idx],
                "rms": X_rms_train[batch_idx],
                "mask": X_mask_train[batch_idx]
            }

            key, subkey = jax.random.split(key, 2)
            
            loss_value, params, ema_params, opt_state = opt_step(
                params, ema_params, opt_state, clean_batch, subkey, activate=activate
            )
            losses.append(loss_value)

        loss_train = np.stack(losses).mean() if losses else 0.0

        losses = []
        for i in range(0, num_test_samples - cfg.batch_size + 1, cfg.batch_size):
            batch_test_idx = np.arange(i, i + cfg.batch_size)
            
            clean_batch_test = {
                "sci_subtracted": X_sci_test[batch_test_idx],
                "psf_stamp": X_psf_test[batch_test_idx],
                "rms": X_rms_test[batch_test_idx],
                "mask": X_mask_test[batch_test_idx]
            }

            subkey, key = jax.random.split(key, 2)
            loss_value = loss(params, clean_batch_test, subkey, 0.0)
            losses.append(loss_value)

        loss_test = np.stack(losses).mean() if losses else 0.0

        if (epoch + 1) % cfg.log_freq == 0:
            model = eqx.combine(params, static)
            model = eqx.nn.inference_mode(model, True)
            y, _, _ = jax.vmap(model)(clean_batch["sci_subtracted"], clean_batch["psf_stamp"])

            x = plot_ae_residuals(clean_batch, y)

            plot_path = exp_path / f"residuals_epoch_{epoch+1}.png"
            hist_path = exp_path / f"hist_residuals_epoch_{epoch+1}.png"
            plot_residual_histogram(clean_batch, y, path=hist_path)


            run.log({
                "loss_train": loss_train,
                "loss_test": loss_test,
                "fit_and_residuals": wandb.Image(x),
                "histogram_residuals": wandb.Image(str(hist_path)),
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