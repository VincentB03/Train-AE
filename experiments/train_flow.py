#!/usr/bin/env python
import os
import jax
import jax.numpy as jnp
import optax
import equinox as eqx

import numpy as np

from torch.utils.data import DataLoader, Dataset

from pshear.utils import load_galaxy_autoencoder, dump_flow
from pshear.nn.flow import make_latent_flow
from paramax import NonTrainable

from datasets import load_dataset

from experiments.utils import PATH

from einops import rearrange
import matplotlib

import wandb

CONFIG = {
    # frozen autoencoder to encode galaxies into latent codes: must point to a
    # checkpoint produced by train_test_partial.py, i.e.
    # PATH / "runs" / ae_run_dir / f"model_checkpoint_{ae_epoch}.eqx" (+ config.yaml)
    "ae_run_dir": "Student-1_i1pf186a",
    "ae_epoch": 2000,
    # flow (unconditional)
    "flow_type": "RealNVP",
    "flow_layers": 4,
    "latent_dim": [1, 4, 4],
    "cond_dim": None,
    "bijector": "affine",
    "nn_width": 128,
    "nn_depth": 2,
    "knots": 28,
    "interval": 5.5,
    # optimization
    "learning_rate": 1e-3,
    "batch_size": 256,
    "epochs": 1000,
    "log_freq": 10,
}


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
        collate_fn=lambda batch: {
            k: np.stack([b[k] for b in batch]) for k in batch[0]
        },
    )


def preprocess_batch(batch_raw):
    return {
        "sci_subtracted": jnp.expand_dims(batch_raw["sci_subtracted"], axis=1),
        "psf_stamp": jnp.expand_dims(batch_raw["psf_stamp"], axis=1),
    }


def train(runid: str = None):
    run = wandb.init(
        project="pshear-euclid-flow",
        name="flow"+("-" + runid if runid else ""),
        id=runid,
        resume="allow",
        dir=PATH,
        config=CONFIG,
    )

    cfg = run.config

    exp_path = PATH / f"runs/flow/{run.name}_{run.id}"
    exp_path.mkdir(parents=True, exist_ok=True)

    print("Loading Dataset from Hugging Face")
    dset = load_dataset("VincentB03/euclid-Q1-V2", split="train", keep_in_memory=True)
    dset = dset.train_test_split(test_size=0.1, seed=42)
    dset = dset.with_format("numpy")

    train_loader = make_loader(dset["train"], cfg.batch_size, shuffle=True)
    test_loader = make_loader(dset["test"], cfg.batch_size, shuffle=False)

    key = jax.random.key(0)

    # frozen, pretrained autoencoder: never updated, only used to (de)code latents
    ae = load_galaxy_autoencoder(PATH / "runs" / cfg.ae_run_dir, epoch=cfg.ae_epoch)
    ae = eqx.nn.inference_mode(ae, value=True)

    flow = make_latent_flow(
        latent_dim=cfg.latent_dim,
        cond_dim=cfg.cond_dim,
        bijector=cfg.bijector,
        nn_width=cfg.nn_width,
        nn_depth=cfg.nn_depth,
        knots=cfg.knots,
        interval=cfg.interval,
        flow_type=cfg.flow_type,
        flow_layers=cfg.flow_layers,
        key=key,
    )

    params, static = eqx.partition(
        flow,
        eqx.is_inexact_array,
        is_leaf=lambda leaf: isinstance(leaf, NonTrainable),
    )

    @jax.jit
    def loss(params, batch):
        z = jax.vmap(ae.encode)(batch["sci_subtracted"])
        flow = eqx.combine(params, static)
        z = flow.flatten_latent(z)
        log_p = flow.log_prob(z)
        return -log_p.mean()

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=cfg.learning_rate),
    )
    opt_state = optimizer.init(params)

    @jax.jit
    def opt_step(params, opt_state, batch):
        loss_value, grads = jax.value_and_grad(loss)(params, batch)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        return loss_value, params, opt_state

    @jax.jit
    def sample_images(params, batch, key):
        flow = eqx.combine(params, static)
        n = batch["psf_stamp"].shape[0]
        z = flow.sample(key=key, sample_shape=(n,))
        g = jax.vmap(ae.decode)(flow.unflatten_latent(z))
        y = jax.vmap(ae.convolve)(g, batch["psf_stamp"])
        y = y / y.max((-1, -2), keepdims=True)

        return y

    for epoch in range(cfg.epochs):
        print(f"Epoch {epoch + 1}/{cfg.epochs}")

        losses = []
        for batch in train_loader:
            batch = preprocess_batch(batch)
            loss_value, params, opt_state = opt_step(params, opt_state, batch)
            losses.append(loss_value)

        loss_train = np.stack(losses).mean() if losses else 0.0

        losses = []
        for batch in test_loader:
            batch = preprocess_batch(batch)
            loss_value = loss(params, batch)
            losses.append(loss_value)

        loss_val = np.stack(losses).mean() if losses else 0.0

        if (epoch + 1) % cfg.log_freq == 0:
            key, subkey = jax.random.split(key)
            y = sample_images(params, batch, subkey)
            image = rearrange(y[:16], "(b1 b2) c h w -> (b1 h) (b2 c w)", b1=4, b2=4)
            cmap = matplotlib.colormaps["viridis"]
            normed_image = (image - np.min(image)) / (np.max(image) - np.min(image))
            rgba_image = cmap(normed_image)
            x = (rgba_image[..., :3] * 255).astype(np.uint8)

            run.log(
                {
                    "loss_train": loss_train,
                    "loss_val": loss_val,
                    "flow_samples": wandb.Image(x),
                }
            )

            flow = eqx.combine(params, static)
            dump_flow(exp_path, flow, epoch + 1, CONFIG)

            wandb.save(str(exp_path / "*"), base_path=str(exp_path.parent))
        else:
            run.log(
                {
                    "loss_train": loss_train,
                    "loss_val": loss_val,
                }
            )

    wandb.finish()


if __name__ == "__main__":
    runid = wandb.util.generate_id()
    train(runid=runid)
