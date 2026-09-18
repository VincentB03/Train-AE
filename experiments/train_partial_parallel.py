#!/usr/bin/env python
import os
import inspect
import time
import types
import jax
import jax.numpy as jnp
import optax
import equinox as eqx
import numpy as np
import torch
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler, SequentialSampler
from pshear.galaxy import GalaxyAutoEncoderLoss, make_galaxy_autoencoder
from pshear.utils import dump_galaxy_autoencoder

from datasets import load_dataset
# data augmentation: random horizontal/vertical flips (train only)
from experiments.utils import PATH, plot_ae_residuals

import wandb

try:
    from jax import shard_map  # public API in recent JAX
except ImportError:
    from jax.experimental.shard_map import shard_map  # older JAX

# Keyword that turns shard_map's replication check off: it was named check_rep
# in jax.experimental and renamed check_vma when shard_map became public, so
# pick whichever the installed JAX accepts (see the comment on sharded_grads).
try:
    _shard_map_params = inspect.signature(shard_map).parameters
except (TypeError, ValueError):  # pragma: no cover - unusual wrapping
    _shard_map_params = {}
if "check_vma" in _shard_map_params:
    NO_REP_CHECK = {"check_vma": False}
elif "check_rep" in _shard_map_params:
    NO_REP_CHECK = {"check_rep": False}
else:
    NO_REP_CHECK = {}

# Data-parallel version of train_test_partial.py for ONE node with several GPUs
# (Jean Zay V100 quad-GPU node: 4 GPUs).
#
# Single-controller: a single Python process sees and drives every GPU of the
# node, so jax.distributed.initialize() is NOT called and the job must be
# submitted with ONE task. Launching one task per GPU without
# jax.distributed.initialize() would silently start independent trainings that
# never synchronise their gradients -- a guard in train() refuses to run then.
#
# Strategy (Mesh + NamedSharding + jax.jit, no pmap):
#   - model params, EMA params and optimizer state are replicated on every GPU
#     (NamedSharding(mesh, P()));
#   - `batch_size` is the GLOBAL batch: the DataLoader builds it once in host
#     RAM, then shard_batch() sends each GPU only its batch_size // num_devices
#     examples (NamedSharding(mesh, P("data")));
#   - the forward/backward pass runs inside shard_map, so each GPU computes the
#     loss on its own slice only, whatever the XLA partitioner thinks of the
#     jax-galsim FFT convolution; gradients and loss are then averaged with
#     pmean. Shards are equal-sized, so the mean of the per-GPU means is the
#     global batch mean.
#   - 128 examples per GPU: the global batch (512 with 4 GPUs) is 4x the one of
#     train_test_partial.py, i.e. 4x fewer optimizer steps per epoch. Starting
#     point, to tune empirically: learning rates x sqrt(4) = x2, Adam betas
#     kept, EMA decay 0.999**4 (same averaging horizon in epochs).
#   - per-example RNG keys (flips, dropout) are drawn for the global batch and
#     then sharded, exactly like in train_test_partial.py, so two GPUs never
#     reuse the same random draws.
CONFIG = {
    "use_jax_galsim": True,
    "minimum_fft_size": 128,
    "nx": 64,
    "ny": 64,
    "scale": 0.1,  # arcsec/pixel — Euclid VIS pixel scale is 0.1 arcsec/pixel
    # The encoder sees asinh(sci_subtracted / asinh_scale) instead of the raw
    # flux (see GalaxyAutoEncoder.encode). Only the encoder input changes: the
    # loss still compares the raw image to the PSF-convolved decoder output,
    # weighted by the raw noise_map. None: raw flux input.
    # asinh(x/s) is linear for |x| << s and logarithmic for |x| >> s, so s is
    # the typical pixel noise sigma: noise stays in the linear regime, only the
    # bright parts are compressed. s is one fixed value (not a per-stamp or
    # per-pixel rms): the decoder must return absolute fluxes without seeing s,
    # and a per-pixel scale would distort the image seen by the encoder.
    # How it was measured:
    #   rms = dset_train[:5000]["noise_map"]           # (N, 64, 64), shuffled split
    #   s = np.median(np.median(rms, axis=(1, 2)))     # median of per-stamp medians
    # cross-check with the data itself (stamps are mostly background):
    #   x = dset_train[:5000]["sci_subtracted"]
    #   sigma = 1.4826 * np.median(np.abs(x - np.median(x, axis=(1, 2), keepdims=True)), axis=(1, 2))
    #   np.median(sigma)
    # euclid-Q1-VF, first parquet shard (8368 stamps): per-stamp noise medians
    # 2.72-3.36 (1st-99th percentile), s = 2.81, MAD cross-check 2.89. Encoder
    # input goes from [-8, 4e4] (raw) to [-1.8, 10.2] (99.9th percentile 4.8).
    "asinh_scale": 2.8,
    "in_channels": 1,
    "latent_channels": 1,
    "hid_channels": (32, 32, 64, 128, 256),
    "hid_blocks": (2, 2, 2, 2, 2),
    "attention_heads": {4: 4},  # deepest encoder stage (4x4 feature map)
    "patch_size": 1,
    "stride": 2,
    "dropout": 0.05,
    "kernel_size": 3,
    "batch_size": 512,     # GLOBAL batch = 128 per GPU x 4 GPUs, must be divisible by num_devices
    "epochs": 2000,
    # 128-batch learning rates x sqrt(512 / 128) = x2 (square-root scaling rule
    # for Adam, Adam betas kept): starting point, to tune empirically
    "init_learning_rate": 2e-6,
    "peak_learning_rate": 2e-5,
    "end_learning_rate": 2e-7,
    "warmup_epochs": 100,
    "lr_decay_epochs": 300,  # LR reaches end_learning_rate here, then holds flat
    "weight_decay": 1e-4,
    # 0.999 at batch 128; beta**4 keeps the same averaging horizon in epochs
    # with 4x fewer steps per epoch
    "ema_decay": 0.999 ** 4,
    "losses": ["student_t_masked"],
    "weights": [1.0],
    "log_freq": 10,
    "num_devices": 4,      # GPUs this process must see; None accepts any count
}


def ema_update(params, ema_params, decay):
    return jax.tree_util.tree_map(
        lambda p, e: decay * e + (1.0 - decay) * p, params, ema_params
    )


# Fields that live on the same pixel grid and must receive the same flip.
AUGMENT_KEYS = ("sci_subtracted", "psf_stamp", "noise_map", "binary_mask")


def random_flip(x, key, axis):
    do_flip = jax.random.bernoulli(key)
    return jnp.where(do_flip, jnp.flip(x, axis=axis), x)


def augment_single(example, key):
    # example[k] has shape (H, W); the same key is used for every field so
    # image, PSF, noise map and mask stay consistent with each other
    keys = jax.random.split(key, 2)
    out = dict(example)
    for k in AUGMENT_KEYS:
        out[k] = random_flip(out[k], keys[0], -1)
        out[k] = random_flip(out[k], keys[1], -2)
    return out


augment_batch = jax.vmap(augment_single)


# dataset column -> batch field
COLUMNS = {
    "sci_subtracted": "sci_subtracted",
    "psf_residual": "psf_stamp",
    "noise_map": "noise_map",
    "binary_mask": "binary_mask",
}


# Dtype each field is narrowed to inside the worker, before the batch travels
# through shared memory to the main process. binary_mask is stored as int64,
# 8 bytes per pixel for a 0/1 flag, and is half of a batch's bytes on its own;
# as bool it is 8x smaller. JAX widens it back on device (nll * mask promotes
# to float32, mask.sum() counts the same).
BATCH_DTYPES = {"binary_mask": np.bool_}


def as_batch(column, dtype=None):
    # with_format("numpy") gives one (B, H, W) array when all rows share a
    # shape, otherwise an object array of (H, W) arrays
    column = np.asarray(column)
    batch = np.stack(column) if column.dtype == object else column
    return batch if dtype is None else batch.astype(dtype, copy=False)


class HFDataset(Dataset):
    # Indexed with a whole list of indices (see make_loader): one batched Arrow
    # lookup per batch instead of batch_size single-row lookups + np.stack, so
    # the loader can keep up with 4 GPUs.
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset.select_columns(list(COLUMNS))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        items = self.dataset[indices]
        return {
            field: as_batch(items[column], BATCH_DTYPES.get(field))
            for column, field in COLUMNS.items()
        }


def identity(batch):
    # module-level (not a lambda): picklable if workers are spawned instead of
    # forked (macOS, Python >= 3.14 on Linux)
    return batch


def make_loader(hf_dataset, batch_size, shuffle=False, seed=0):
    dataset = HFDataset(hf_dataset)
    if shuffle:
        # seeded generator: reproducible data order, new permutation every epoch
        sampler = RandomSampler(dataset, generator=torch.Generator().manual_seed(seed))
    else:
        sampler = SequentialSampler(dataset)
    return DataLoader(
        dataset,
        # the sampler yields whole batches of indices, so automatic batching is
        # disabled (batch_size=None) and each worker builds a full global batch
        sampler=BatchSampler(sampler, batch_size=batch_size, drop_last=True),
        batch_size=None,
        collate_fn=identity,
        # one worker per CPU of the task, capped: the main process also needs
        # CPU to dispatch the GPU work
        num_workers=min(int(os.environ.get("SLURM_CPUS_PER_TASK", 4)), 16),
        prefetch_factor=2,
        pin_memory=False,
        persistent_workers=True,
    )


def shard_batch(batch, sharding):
    # host (numpy) global batch -> global jax.Array split along its first axis:
    # each GPU only receives its own slice, the full batch is never put on a
    # single device
    return jax.tree_util.tree_map(
        lambda x: jax.make_array_from_process_local_data(sharding, x), batch
    )


def train(runid: str):
    # number of launched copies of this script: srun (SLURM_*), mpirun (OMPI_*, PMI_*)
    n_tasks = max(
        int(os.environ.get(var, 1))
        for var in ("SLURM_NTASKS", "SLURM_STEP_NUM_TASKS", "OMPI_COMM_WORLD_SIZE", "PMI_SIZE")
    )
    if n_tasks > 1 and jax.process_count() == 1:
        raise RuntimeError(
            f"{n_tasks} tasks launched: this script is single-controller and must run as ONE "
            "process that sees all the GPUs of the node (--ntasks=1, no mpirun -np N). With one "
            "process per GPU and no jax.distributed.initialize(), each would train its own model."
        )

    devices = jax.devices()
    num_devices = len(devices)
    is_main = jax.process_index() == 0
    if is_main:
        print(
            f"JAX {jax.__version__}, process {jax.process_index()}/{jax.process_count()}, "
            f"{num_devices} device(s): {devices}"
        )
    if CONFIG["num_devices"] is not None and num_devices != CONFIG["num_devices"]:
        raise RuntimeError(
            f"Expected {CONFIG['num_devices']} devices, JAX sees {num_devices} ({devices}). "
            "Check --gres=gpu:N, CUDA_VISIBLE_DEVICES and that jax was installed with CUDA "
            "support (a CPU-only jax sees a single CPU device)."
        )
    assert CONFIG["batch_size"] % num_devices == 0, (
        f"batch_size {CONFIG['batch_size']} must be divisible by the number of "
        f"devices {num_devices}."
    )

    # Same config on every process (not run.config, which only exists where
    # wandb.init was called).
    cfg = types.SimpleNamespace(**CONFIG)

    # W&B and checkpoints only on process 0. Single-controller means there is
    # only one process anyway, the guard is kept as a safety net.
    if is_main:
        run = wandb.init(
            project="Test-AE-partial-3-parallel",
            name="Student-2-parallel",
            id=runid,
            resume="allow",
            dir=PATH,
            config=CONFIG,
        )
        exp_path = PATH / f"runs/{run.name}_{run.id}"
        exp_path.mkdir(parents=True, exist_ok=True)

        print("Loading Dataset from Hugging Face")
    dset = load_dataset("VincentB03/euclid-Q1-VF", split="train", keep_in_memory=True)  # Try keeping in memory for faster training

    dset = dset.train_test_split(test_size=0.1, seed=42)
    dset = dset.with_format("numpy")
    dset_train = dset["train"]
    dset_test = dset["test"]

    train_loader = make_loader(dset_train, cfg.batch_size, shuffle=True)
    test_loader = make_loader(dset_test, cfg.batch_size, shuffle=False)

    # 1-D mesh over all the GPUs: "data" is the batch axis
    mesh = Mesh(np.array(devices), axis_names=("data",))
    replicated = NamedSharding(mesh, P())
    data_sharding = NamedSharding(mesh, P("data"))

    key = jax.random.PRNGKey(0)

    model = make_galaxy_autoencoder(
        use_jax_galsim=cfg.use_jax_galsim, minimum_fft_size=cfg.minimum_fft_size,
        nx=cfg.nx, ny=cfg.ny, scale=cfg.scale, in_channels=cfg.in_channels,
        latent_channels=cfg.latent_channels, hid_channels=cfg.hid_channels,
        hid_blocks=cfg.hid_blocks, attention_heads=cfg.attention_heads,
        patch_size=cfg.patch_size, stride=cfg.stride, dropout=cfg.dropout,
        kernel_size=cfg.kernel_size, asinh_scale=cfg.asinh_scale, key=key,
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
            "mask": mask,
        }

    # `keys` holds one key per example of the batch the function receives
    def loss(params, batch, keys, activate):
        batch = preprocess_batch(batch)
        model = eqx.combine(params, static)
        return loss_fn(
            model,
            batch["sci_subtracted"],
            batch["psf_stamp"],
            batch["rms"],
            batch["mask"],
            keys, activate
        ).mean()

    steps_per_epoch = len(train_loader)
    warmup_steps = cfg.warmup_epochs * steps_per_epoch
    decay_steps = cfg.lr_decay_epochs * steps_per_epoch

    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=cfg.init_learning_rate,
        peak_value=cfg.peak_learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        end_value=cfg.end_learning_rate,
    )

    def weight_decay_mask(params):
        # apply weight decay to weight matrices, not to biases
        is_bias = lambda path: any(
            isinstance(p, jax.tree_util.GetAttrKey) and p.name == "bias" for p in path
        )
        return jax.tree_util.tree_map_with_path(lambda path, x: not is_bias(path), params)

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(
            learning_rate=lr_schedule, b1=0.9, b2=0.95,
            weight_decay=cfg.weight_decay, mask=weight_decay_mask,
        ),
    )
    opt_state = optimizer.init(params)

    # full copy of the model state on every GPU
    params, ema_params, opt_state = jax.device_put((params, ema_params, opt_state), replicated)

    # --- per-device functions: run on ONE GPU with its local slice ----------
    def local_grads(params, batch, aug_keys, keys, activate):
        batch = augment_batch(batch, aug_keys)
        loss_value, grads = jax.value_and_grad(loss)(params, batch, keys, activate)
        # the only communication of the step: all-reduce of loss and gradients
        return jax.lax.pmean((loss_value, grads), axis_name="data")

    def local_test_loss(ema_params, batch, keys, activate):
        model = eqx.nn.inference_mode(eqx.combine(ema_params, static), value=True)
        batch = preprocess_batch(batch)
        loss_value = loss_fn(
            model,
            batch["sci_subtracted"],
            batch["psf_stamp"],
            batch["rms"],
            batch["mask"],
            keys, activate
        ).mean()
        return jax.lax.pmean(loss_value, axis_name="data")

    def local_predict(params, img, psf):
        model = eqx.nn.inference_mode(eqx.combine(params, static), value=True)
        y, _, _ = jax.vmap(model)(img, psf)
        return y

    # NO_REP_CHECK: with the check left on (the default), JAX >= 0.11 tags the
    # arrays inside shard_map with "varying manual axes" ({V:data}). jax-galsim
    # calls equinox.error_if, a lax.cond whose two branches then have mismatched
    # types (one varying, one not), which raises at trace time. Turning the
    # check off removes the tagging; pmean still replicates its outputs, but JAX
    # no longer verifies that out_specs=P() really is replicated.
    sharded_grads = shard_map(
        local_grads, mesh=mesh,
        in_specs=(P(), P("data"), P("data"), P("data"), P()),
        out_specs=(P(), P()), **NO_REP_CHECK,
    )
    sharded_test_loss = shard_map(
        local_test_loss, mesh=mesh,
        in_specs=(P(), P("data"), P("data"), P()),
        out_specs=P(), **NO_REP_CHECK,
    )
    sharded_predict = shard_map(
        local_predict, mesh=mesh,
        in_specs=(P(), P("data"), P("data")),
        out_specs=P("data"), **NO_REP_CHECK,
    )

    # --- global steps: jit with explicit input/output shardings ----------------
    # Arguments are passed positionally (jit with in_shardings rejects kwargs).
    def opt_step(params, ema_params, opt_state, batch, key, activate):
        # same key derivation as train_test_partial.py, over the GLOBAL batch
        batch_size = batch["sci_subtracted"].shape[0]
        key, aug_key = jax.random.split(key)
        aug_keys = jax.random.split(aug_key, batch_size)
        keys = jax.random.split(key, batch_size)
        loss_value, grads = sharded_grads(params, batch, aug_keys, keys, activate)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        ema_params = ema_update(params, ema_params, decay=cfg.ema_decay)
        return loss_value, params, ema_params, opt_state

    opt_step = jax.jit(
        opt_step,
        in_shardings=(replicated, replicated, replicated, data_sharding, replicated, replicated),
        out_shardings=(replicated, replicated, replicated, replicated),
    )

    def test_step(ema_params, batch, key, activate):
        keys = jax.random.split(key, batch["sci_subtracted"].shape[0])
        return sharded_test_loss(ema_params, batch, keys, activate)

    test_step = jax.jit(
        test_step,
        in_shardings=(replicated, data_sharding, replicated, replicated),
        out_shardings=replicated,
    )

    predict = jax.jit(
        sharded_predict,
        in_shardings=(replicated, data_sharding, data_sharding),
        out_shardings=data_sharding,
    )

    activate = jnp.array(0.0)
    for epoch in range(cfg.epochs):
        if is_main:
            print(f"Epoch {epoch+1}/{cfg.epochs}")

        # JAX dispatches GPU work asynchronously: the loop only blocks in the
        # DataLoader (measured as data_wait) and at float(), which waits for the
        # GPUs. A data_wait_frac close to 1 means the GPUs are starved by the loader.
        epoch_start = time.perf_counter()
        data_wait = 0.0
        losses = []
        t = time.perf_counter()
        for batch in train_loader:
            data_wait += time.perf_counter() - t
            key, subkey = jax.random.split(key, 2)
            loss_value, params, ema_params, opt_state = opt_step(
                params, ema_params, opt_state, shard_batch(batch, data_sharding), subkey, activate
            )
            losses.append(loss_value)  # no host sync inside the loop
            t = time.perf_counter()

        loss_train = float(jnp.stack(losses).mean()) if losses else 0.0
        train_time = time.perf_counter() - epoch_start
        train_samples = len(losses) * cfg.batch_size

        losses = []
        for batch in test_loader:
            key, subkey = jax.random.split(key, 2)
            loss_value = test_step(ema_params, shard_batch(batch, data_sharding), subkey, activate)
            losses.append(loss_value)

        loss_test = float(jnp.stack(losses).mean()) if losses else 0.0

        # LR used at the last optimizer step of this epoch
        learning_rate = float(lr_schedule((epoch + 1) * steps_per_epoch - 1))

        metrics = {
            "loss_train": loss_train,
            "loss_test": loss_test,
            "learning_rate": learning_rate,
            "epoch_time_s": train_time,
            "train_samples_per_s": train_samples / train_time,
            "data_wait_frac": data_wait / train_time,
        }

        if is_main:
            print(
                f"  epoch {epoch + 1}: "
                + "  ".join(f"{k}={v:.5g}" for k, v in metrics.items()),
                flush=True,
            )

        if (epoch + 1) % cfg.log_freq == 0:
            # computed on every process (it is a collective when there are
            # several), logged/saved by process 0 only
            img = np.expand_dims(batch["sci_subtracted"], axis=1)
            psf = np.expand_dims(batch["psf_stamp"], axis=1)
            y = predict(params, shard_batch(img, data_sharding), shard_batch(psf, data_sharding))

            if is_main:
                x = plot_ae_residuals({"sci_subtracted": img}, np.asarray(y))
                metrics["fit_and_residuals"] = wandb.Image(x)
                run.log(metrics)

                # replicated arrays are read back from one GPU when serialised
                model = eqx.nn.inference_mode(eqx.combine(params, static), True)
                dump_galaxy_autoencoder(exp_path, model, epoch + 1, CONFIG)
                wandb.save(str(exp_path / "*"), base_path=str(exp_path.parent))
        elif is_main:
            run.log(metrics)

    if is_main:
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
