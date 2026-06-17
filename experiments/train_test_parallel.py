#!/usr/bin/env python

import os
import functools # [NOUVEAU] Nécessaire pour configurer pmap
import jax
import jax.numpy as jnp
import optax
import equinox as eqx
import numpy as np
import matplotlib.pyplot as plt

from pshear.galaxy import GalaxyAutoEncoderLoss, make_galaxy_autoencoder
from pshear.utils import dump_galaxy_autoencoder
from datasets import load_dataset
from experiments.utils import PATH, plot_ae_residuals
import wandb

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
    "epochs": 2,        
    "learning_rate": 1e-5,
    "losses": ["mae", "tv"],
    "weights": [1.0, 0.0001],
    "log_freq": 1,
}

def ema_update(params, ema_params, decay):
    return jax.tree_util.tree_map(
        lambda p, e: decay * e + (1.0 - decay) * p, params, ema_params
    )

def replicate(tree, devices):
    return jax.device_put_replicated(tree, devices)

def unreplicate(tree):
    return jax.tree_util.tree_map(lambda x: x[0], tree)

def shard_batch(batch, num_devices):
    return jax.tree_util.tree_map(
        lambda x: x.reshape((num_devices, x.shape[0] // num_devices) + x.shape[1:]), 
        batch
    )

def train(runid: str):
    num_devices = jax.local_device_count()
    devices = jax.local_devices()
    print(f"Launched on {num_devices} devices")
    
    assert CONFIG["batch_size"] % num_devices == 0, f"Batch size {CONFIG['batch_size']} must be divisible by number of devices {num_devices}."

    run = wandb.init(
        project="Test_AE",
        name="parallel-test",
        id=runid,
        resume="allow",
        dir=PATH,
        config=CONFIG,
    )

    exp_path = PATH / f"runs/{run.name}_{run.id}"
    exp_path.mkdir(parents=True, exist_ok=True)
    cfg = run.config
    
    print("Loading Dataset from Hugging Face")
    dset = load_dataset("VincentB03/Toy-Dataset-COSMOS", split="train", keep_in_memory=True) 
    
    dset = dset.train_test_split(test_size=0.1, seed=42)
    dset = dset.with_format("numpy")
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
        in_axes=(None, 0, 0, 0, 0, 0, None),
    )

    def preprocess_batch(batch_raw):
        img = jnp.expand_dims(batch_raw["sci_subtracted"], axis=1)
        psf = jnp.expand_dims(batch_raw["psf_stamp"], axis=1)
        rms = jnp.expand_dims(batch_raw["noise_map"], axis=1)
        mask = jnp.expand_dims(batch_raw["binary_mask"], axis=1)
        return {
            "sci_subtracted": img,
            "psf_stamp": psf,
            "rms": rms,
            "mask": mask
        }

    def loss(params, batch, key, activate):
        batch = preprocess_batch(batch)
        model = eqx.combine(params, static)
        batch_size = batch["sci_subtracted"].shape[0] # Correspondra maintenant à batch_size // num_devices
        keys = jax.random.split(key, batch_size)
        return loss_fn(
            model, 
            batch["sci_subtracted"], 
            batch["psf_stamp"], 
            batch["rms"], 
            batch["mask"],
            keys, activate
        ).mean()

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=cfg.learning_rate, b1=0.9, b2=0.95, weight_decay=1e-4),
    )
    opt_state = optimizer.init(params)

    params = replicate(params, devices)
    ema_params = replicate(ema_params, devices)
    opt_state = replicate(opt_state, devices)

    @functools.partial(jax.pmap, axis_name="devices", in_axes=(0, 0, 0, 0, 0, None))
    def opt_step(params, ema_params, opt_state, batch, key, activate=0.0):
        loss_value, grads = jax.value_and_grad(loss)(params, batch, key, activate)
        
        grads = jax.lax.pmean(grads, axis_name="devices")
        loss_value = jax.lax.pmean(loss_value, axis_name="devices")
        
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        ema_params = ema_update(params, ema_params, decay=0.999)
        return loss_value, params, ema_params, opt_state

    @functools.partial(jax.pmap, axis_name="devices", in_axes=(0, 0, 0, None))
    def test_loss_step(ema_params, batch, key, activate):
        batch = preprocess_batch(batch)
        model = eqx.combine(ema_params, static)
        model = eqx.nn.inference_mode(model, value=True)
        batch_size = batch["sci_subtracted"].shape[0]
        keys = jax.random.split(key, batch_size)
        loss_value = loss_fn(
            model, 
            batch["sci_subtracted"], 
            batch["psf_stamp"], 
            batch["rms"], 
            batch["mask"],
            keys, 
            activate
        ).mean()
        return jax.lax.pmean(loss_value, axis_name="devices")

    activate = 0.0
    for epoch in range(cfg.epochs):
        print(f"Epoch {epoch+1}/{cfg.epochs}")
        loader = dset_train.shuffle(seed=epoch).iter(batch_size=cfg.batch_size, drop_last_batch=True)

        if epoch == 100: 
            activate = 1.0
            
        losses = []
        for batch in loader:
            key, subkey = jax.random.split(key, 2)
            
            clean_batch = {
                "sci_subtracted": batch["sci_subtracted"],
                "psf_stamp": batch["psf_stamp"],
                "noise_map": batch["noise_map"],
                "binary_mask": batch["binary_mask"]
            }
            
            sharded_batch = shard_batch(clean_batch, num_devices)
            step_keys = jax.random.split(subkey, num_devices)

            loss_value, params, ema_params, opt_state = opt_step(
                params, ema_params, opt_state, sharded_batch, step_keys, activate
            )
            losses.append(np.array(loss_value[0]))

        loss_train = np.stack(losses).mean() if losses else 0.0

        loader = dset_test.iter(batch_size=cfg.batch_size, drop_last_batch=True)
        losses = []
        for batch in loader:
            key, subkey = jax.random.split(key, 2)
            clean_batch = {
                "sci_subtracted": batch["sci_subtracted"],
                "psf_stamp": batch["psf_stamp"],
                "noise_map": batch["noise_map"],
                "binary_mask": batch["binary_mask"]
            }
            
            sharded_batch = shard_batch(clean_batch, num_devices)
            step_keys = jax.random.split(subkey, num_devices)

            loss_value = test_loss_step(ema_params, sharded_batch, step_keys, activate)
            losses.append(np.array(loss_value[0]))

        loss_test = np.stack(losses).mean() if losses else 0.0

        if (epoch + 1) % cfg.log_freq == 0:
            single_params = unreplicate(params)
            model = eqx.combine(single_params, static)
            model = eqx.nn.inference_mode(model, True)
            
            plot_batch = preprocess_batch(batch)
            y, _, _ = jax.vmap(model)(plot_batch["sci_subtracted"], plot_batch["psf_stamp"])

            x = plot_ae_residuals(plot_batch, y)

            run.log({
                "loss_train": loss_train,
                "loss_test": loss_test,
                "fit_and_residuals": wandb.Image(x),
            })

            dump_galaxy_autoencoder(exp_path, model, epoch + 1, CONFIG)
            wandb.save(str(exp_path / "*"), base_path=str(exp_path.parent))
        else:
            run.log({"loss_train": loss_train, "loss_test": loss_test})

    model_final = eqx.combine(unreplicate(params), static)
    
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