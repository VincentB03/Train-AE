#!/usr/bin/env python
import os
import jax
import jax.numpy as jnp
import optax
import equinox as eqx
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
from pshear.galaxy import GalaxyAutoEncoderLoss, make_galaxy_autoencoder
from pshear.utils import dump_galaxy_autoencoder

from datasets import load_dataset
# no data augmentation 
from experiments.utils import PATH, plot_ae_residuals

import wandb

CONFIG = {
    "use_jax_galsim": True,
    "minimum_fft_size": 64, 
    "nx": 64,
    "ny": 64,
    "scale": 0.1,  # arcsec/pixel — Euclid VIS pixel scale is 0.1 arcsec/pixel
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
    "epochs": 2000,        
    "learning_rate": 1e-6,
    "epoch_to_decay": [400, 800, 1200, 1600],  # epochs at which the LR is decayed
    "lr_decay_factor": 0.5,  # scale applied at each decay epoch (single float, or a list matching epoch_to_decay)
    "losses": ["chi2_masked"],
    "weights": [1.0],
    "log_freq": 10,
}


def ema_update(params, ema_params, decay):
    return jax.tree_util.tree_map(
        lambda p, e: decay * e + (1.0 - decay) * p, params, ema_params
    )
class HFDataset(Dataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            "sci_subtracted": item["sci_subtracted"],
            "psf_stamp": item["psf_residual"],
            "noise_map": item["noise_map"],
            "binary_mask": item["binary_mask"],
        }

def make_loader(hf_dataset, batch_size, shuffle=False):
    return DataLoader(
        HFDataset(hf_dataset),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)),
        prefetch_factor=2,
        pin_memory=False,
        drop_last=True,
        persistent_workers=True,
        collate_fn=lambda batch: {k: np.stack([b[k] for b in batch]) for k in batch[0]},
    )

def train(runid: str):
    run = wandb.init(
        project="Test-AE-partial-2",
        name="CHI2_MASKED_FULL_WEIGHTS",
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
    dset = dset.with_format("numpy")
    dset_train = dset["train"]
    dset_test = dset["test"]

    train_loader = make_loader(dset_train, cfg.batch_size, shuffle=True)
    test_loader  = make_loader(dset_test,  cfg.batch_size, shuffle=False)

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
    @jax.jit
    def loss(params, batch, key, activate):
        batch = preprocess_batch(batch)
        model = eqx.combine(params, static)
        batch_size = batch["sci_subtracted"].shape[0]
        keys = jax.random.split(key, batch_size)
        return loss_fn(
            model, 
            batch["sci_subtracted"], 
            batch["psf_stamp"], 
            batch["rms"], 
            batch["mask"],
            keys, activate
        ).mean()
    @jax.jit
    def test_loss(ema_params, batch, key, activate):
        batch = preprocess_batch(batch)
        model = eqx.combine(ema_params, static)
        model = eqx.nn.inference_mode(model, value=True)
        batch_size = batch["sci_subtracted"].shape[0]
        keys = jax.random.split(key, batch_size)
        return loss_fn(
            model, 
            batch["sci_subtracted"], 
            batch["psf_stamp"], 
            batch["rms"], 
            batch["mask"],
            keys, 
            activate
        ).mean()
    
    steps_per_epoch = len(train_loader)

    decay_epochs = cfg.epoch_to_decay if isinstance(cfg.epoch_to_decay, (list, tuple)) else [cfg.epoch_to_decay]
    decay_factors = cfg.lr_decay_factor if isinstance(cfg.lr_decay_factor, (list, tuple)) else [cfg.lr_decay_factor] * len(decay_epochs)

    boundaries_and_scales = {
        epoch * steps_per_epoch: factor
        for epoch, factor in zip(decay_epochs, decay_factors)
    }

    lr_schedule = optax.piecewise_constant_schedule(
        init_value=cfg.learning_rate,
        boundaries_and_scales=boundaries_and_scales,
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=lr_schedule, b1=0.9, b2=0.95, weight_decay=1e-4),
    )
    opt_state = optimizer.init(params)

    @jax.jit
    def opt_step(params, ema_params, opt_state, batch, key, activate=0.0):
        loss_value, grads = jax.value_and_grad(loss)(params, batch, key, activate)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        ema_params = ema_update(params, ema_params, decay=0.999)
        return loss_value, params, ema_params, opt_state

    activate = jnp.array(0.0)
    for epoch in range(cfg.epochs):
        print(f"Epoch {epoch+1}/{cfg.epochs}")
            
        losses = []
        for batch in train_loader:
            key, subkey = jax.random.split(key, 2)
            loss_value, params, ema_params, opt_state = opt_step(
                params, ema_params, opt_state, batch, subkey, activate=activate
            )
            losses.append(loss_value)

        loss_train = np.stack(losses).mean() if losses else 0.0

        losses = []
        for batch in test_loader:
            key, subkey = jax.random.split(key, 2)
            loss_value = test_loss(ema_params, batch, subkey, activate=activate)
            losses.append(loss_value)

        loss_test = np.stack(losses).mean() if losses else 0.0

        if (epoch + 1) % cfg.log_freq == 0:
            model = eqx.combine(params, static)
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