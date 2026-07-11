#!/usr/bin/env python

import os
import functools
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
    "scale": 0.1, #Euclid VIS pixel scale is 0.1 arcsec/pixel
    "in_channels": 1,
    "latent_channels": 1,
    "hid_channels": (32, 32, 64, 128, 256),
    "hid_blocks": (2, 2, 2, 2, 2),
    "attention_heads": {5: 4},
    "patch_size": 1,
    "stride": 2,
    "dropout": 0.05,
    "kernel_size": 3,
    "batch_size_per_device": 128,  # sample per GPU and per step
    "epochs": 2000,
    "learning_rate": 1e-6,  # base learning rate for a single GPU, uptable to scale linearly with the number of GPUs
    "warmup_epochs": 5,
    "epoch_to_decay": 400,
    "lr_decay_factor": 0.5,
    "losses": ["chi2_masked"],
    "weights": [1.0],
    "log_freq": 10,
}

def ema_update(params, ema_params, decay):
    return jax.tree_util.tree_map(
        lambda p, e: decay * e + (1.0 - decay) * p, params, ema_params
    )

def replicate(tree, devices):
    # Drop-in replacement for the deprecated jax.device_put_replicated.
    # Uses PmapSharding directly (what device_put_replicated built internally)
    # rather than a NamedSharding, which pmap has to reshard on every call and
    # which triggers NCCL failures on multi-GPU nodes.
    def _replicate_leaf(x):
        x = jnp.asarray(x)
        sharding = jax.sharding.PmapSharding.default(
            (len(devices),) + x.shape, sharded_dim=0, devices=devices
        )
        return jax.device_put(jnp.stack([x] * len(devices)), sharding)
    return jax.tree_util.tree_map(_replicate_leaf, tree)

def unreplicate(tree):
    return jax.tree_util.tree_map(lambda x: x[0], tree)

def shard_batch(batch, num_devices):
    return jax.tree_util.tree_map(
        lambda x: x.reshape((num_devices, x.shape[0] // num_devices) + x.shape[1:]),
        batch
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
    num_devices = jax.local_device_count()
    devices = jax.local_devices()
    print(f"Launched on {num_devices} devices")

    run = wandb.init(
        project="Test-AE-partial-2-parallel",
        name="CHI2_MASKED",
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

    # Fixed batch per GPU: the global batch increases with the number of GPUs used.
    global_batch_size = cfg.batch_size_per_device * num_devices
    scaled_learning_rate = cfg.learning_rate * num_devices  # linear scaling rule
    print(
        f"batch_size_per_device={cfg.batch_size_per_device} x {num_devices} devices "
        f"-> global_batch_size={global_batch_size}, learning_rate={cfg.learning_rate} "
        f"-> scaled_learning_rate={scaled_learning_rate}"
    )
    run.config.update(
        {"global_batch_size": global_batch_size, "scaled_learning_rate": scaled_learning_rate},
        allow_val_change=True,
    )

    train_loader = make_loader(dset_train, global_batch_size, shuffle=True)
    test_loader = make_loader(dset_test, global_batch_size, shuffle=False)

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

    steps_per_epoch = len(train_loader)
    decay_step = cfg.epoch_to_decay * steps_per_epoch
    warmup_steps = cfg.warmup_epochs * steps_per_epoch

    # Linear warm-up to the scaled learning rate, then a decaying plateau at epoch_to_decay.
    # See linear scaling rule (Goyal et al., 2017): a global batch num_devices times larger
    # uses a learning rate num_devices times larger, reached gradually to avoid
    # destabilizing the start of training.
    if warmup_steps > 0:
        warmup_schedule = optax.linear_schedule(
            init_value=0.0, end_value=scaled_learning_rate, transition_steps=warmup_steps
        )
        post_warmup_schedule = optax.piecewise_constant_schedule(
            init_value=scaled_learning_rate,
            boundaries_and_scales={
                decay_step - warmup_steps: cfg.lr_decay_factor
            }
        )
        lr_schedule = optax.join_schedules(
            schedules=[warmup_schedule, post_warmup_schedule],
            boundaries=[warmup_steps],
        )
    else:
        lr_schedule = optax.piecewise_constant_schedule(
            init_value=scaled_learning_rate,
            boundaries_and_scales={
                decay_step: cfg.lr_decay_factor
            }
        )

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=lr_schedule, b1=0.9, b2=0.95, weight_decay=1e-4),
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

    activate = jnp.array(0.0)
    for epoch in range(cfg.epochs):
        print(f"Epoch {epoch+1}/{cfg.epochs}")

        losses = []
        for batch in train_loader:
            key, subkey = jax.random.split(key, 2)

            sharded_batch = shard_batch(batch, num_devices)
            step_keys = jax.random.split(subkey, num_devices)

            loss_value, params, ema_params, opt_state = opt_step(
                params, ema_params, opt_state, sharded_batch, step_keys, activate
            )
            losses.append(np.array(loss_value[0]))

        loss_train = np.stack(losses).mean() if losses else 0.0

        losses = []
        for batch in test_loader:
            key, subkey = jax.random.split(key, 2)

            sharded_batch = shard_batch(batch, num_devices)
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
