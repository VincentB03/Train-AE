#!/usr/bin/env python

import os
import jax
import optax
import equinox as eqx
import numpy as np
import matplotlib.pyplot as plt

from pshear.galaxy import GalaxyAutoEncoderLoss, make_galaxy_autoencoder
from pshear.utils import dump_galaxy_autoencoder

from datasets import load_dataset
# no data augmentation 
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
    "batch_size": 8,    
    "epochs": 100,        
    "learning_rate": 1e-5,
    "losses": ["mae", "tv"],
    "weights": [1.0, 1e-5],
}


def ema_update(params, ema_params, decay):
    return jax.tree_util.tree_map(
        lambda p, e: decay * e + (1.0 - decay) * p, params, ema_params
    )

def train(runid: str):
    run = wandb.init(
        project="Generative-Euclid",
        name="ae-test-Q1",
        id=runid,
        resume="allow",
        dir=PATH,
        config=CONFIG,
    )

    exp_path = PATH / f"runs/{run.name}_{run.id}"
    exp_path.mkdir(parents=True, exist_ok=True)
    cfg = run.config

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("No HF token(HF_TOKEN).")
    
    print("Loading Dataset from Hugging Face")
    dset = load_dataset("VincentB03/euclid-Q1-V2", split="train", token=hf_token)
    
    dset = dset.train_test_split(test_size=0.1, seed=42)
    
    dset_train = dset["train"].select(range(min(64, len(dset["train"])))).with_format("numpy")
    dset_test = dset["test"].select(range(min(16, len(dset["test"])))).with_format("numpy")

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
        in_axes=(None, 0, 0, None, None),
    )

    @jax.jit
    def loss(params, batch, key, activate):
        model = eqx.combine(params, static)
        return loss_fn(model, batch["sci_subtracted"], batch["psf_stamp"], key, activate).mean()

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

    activate = 0.0
    for epoch in tqdm(range(cfg.epochs)):
        loader = dset_train.shuffle(seed=epoch).iter(batch_size=cfg.batch_size, drop_last_batch=True)

        if epoch == 50: 
            activate = 1.0
            
        losses = []
        for batch in loader:
            batch["sci_subtracted"] = np.expand_dims(batch["sci_subtracted"], axis=1)
            batch["psf_stamp"] = np.expand_dims(batch["psf_stamp"], axis=1)
            key, subkey = jax.random.split(key, 2)
            
            
            loss_value, params, ema_params, opt_state = opt_step(
                params, ema_params, opt_state, batch, subkey, activate=activate
            )
            losses.append(loss_value)

        loss_train = np.stack(losses).mean() if losses else 0.0

        loader = dset_test.iter(batch_size=cfg.batch_size, drop_last_batch=True)
        losses = []
        for batch in loader:
            subkey, key = jax.random.split(key, 2)
            loss_value = loss(params, batch, subkey, 0.0)
            losses.append(loss_value)

        loss_test = np.stack(losses).mean() if losses else 0.0

        if (epoch + 1) % 1 == 0:
            model = eqx.combine(params, static)
            model = eqx.nn.inference_mode(model, True)
            y, _, _ = jax.vmap(model)(batch["sci_subtracted"], batch["psf_stamp"])

            x = plot_ae_residuals(batch, y)

            plot_path = exp_path / f"residuals_epoch_{epoch+1}.png"

            run.log({
                "loss_train": loss_train,
                "loss_test": loss_test,
                "fit_and_residuals": wandb.Image(x),
            })

            dump_galaxy_autoencoder(exp_path, model, epoch + 1, CONFIG)
        else:
            run.log({"loss_train": loss_train, "loss_test": loss_test})

if __name__ == "__main__":
    runid = wandb.util.generate_id()
    train(runid=runid)