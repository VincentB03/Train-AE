from pathlib import Path

import yaml
import equinox as eqx
from .galaxy import make_galaxy_autoencoder
from .nn.flow import make_latent_flow
from jax.random import key

def dump_galaxy_autoencoder(model_path, model, epoch, config):
    with open(model_path / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    eqx.tree_serialise_leaves(model_path / f"model_checkpoint_{epoch}.eqx", model)

def load_galaxy_autoencoder(model_path, epoch=None):
    with open(model_path / "config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    model = make_galaxy_autoencoder(key=key(0), **config)
    if epoch is None:
        checkpoint_path = model_path / "model_checkpoint.eqx"
    else:
        checkpoint_path = model_path / f"model_checkpoint_{epoch}.eqx"
    model = eqx.tree_deserialise_leaves(checkpoint_path, model)
    return model

def dump_flow(model_path, model, epoch, config):
    with open(model_path / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    eqx.tree_serialise_leaves(model_path / f"model_checkpoint_{epoch}.eqx", model)

def load_flow(model_path, epoch=None):
    with open(model_path / "config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    flow = make_latent_flow(key=key(0), **config)
    if epoch is None:
        checkpoint_path = model_path / "model_checkpoint.eqx"
    else:
        checkpoint_path = model_path / f"model_checkpoint_{epoch}.eqx"
    flow = eqx.tree_deserialise_leaves(checkpoint_path, flow)
    return flow

def _unwrap_wandb_config(cfg):
    """Flattens wandb's {desc: null, value: X} config format to X per key,
    and coerces string-encoded literals (kernel_size, nested dict keys)
    back to their Python types."""
    import ast

    result = {}
    for k, v in cfg.items():
        if isinstance(v, dict) and set(v.keys()) <= {"desc", "value"}:
            v = v["value"]
        if isinstance(v, dict):
            try:
                v = {int(kk): vv for kk, vv in v.items()}
            except (ValueError, TypeError):
                pass
        if isinstance(v, str):
            try:
                v = ast.literal_eval(v)
            except (ValueError, SyntaxError):
                pass
        result[k] = v
    return result

def fetch_wandb_checkpoint(run_path, epoch, cache_dir="wandb_weights"):
    """
    Downloads (and caches under `cache_dir`) a run's config.yaml and
    model_checkpoint_<epoch>.eqx from Weights & Biases -- the layout
    written by `dump_galaxy_autoencoder`/`dump_flow` -- and returns the
    local `epoch_dir` containing both files, so `load_galaxy_autoencoder`/
    `load_flow` can read it directly.

    If both files are already cached (e.g. pre-downloaded on a machine
    with internet access and copied over, or a compute node with no
    network access at all), the WandB API is skipped entirely.

    `run_path` is wandb's "entity/project/run_id" identifier.
    """
    import shutil

    epoch = int(epoch)
    checkpoint_fname = "model_checkpoint_%d.eqx" % epoch
    run_root_dir = Path(cache_dir) / run_path.rsplit("/", 1)[-1]
    epoch_dir = run_root_dir / ("epoch_%d" % epoch)
    epoch_dir.mkdir(parents=True, exist_ok=True)

    config_path = run_root_dir / "config.yaml"
    checkpoint_path = epoch_dir / checkpoint_fname

    if not (config_path.exists() and checkpoint_path.exists()):
        import wandb

        api = wandb.Api()
        run = api.run(run_path)
        for file in run.files():
            remote_basename = Path(file.name).name
            if remote_basename not in {"config.yaml", checkpoint_fname}:
                continue
            dest = (
                run_root_dir / remote_basename
                if remote_basename == "config.yaml"
                else epoch_dir / remote_basename
            )
            if dest.exists():
                continue
            downloaded = file.download(root=str(dest.parent), replace=True)
            downloaded_path = Path(downloaded.name)
            if downloaded_path != dest:
                downloaded_path.rename(dest)

    missing = [p for p in (config_path, checkpoint_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing files: %s (check that run %r contains them, or that "
            "cache_dir points at a pre-populated cache when offline)"
            % (", ".join(str(p) for p in missing), run_path)
        )

    shutil.copy(config_path, epoch_dir / "config.yaml")
    patched_config_path = epoch_dir / "config.yaml"
    with open(patched_config_path) as f:
        cfg = _unwrap_wandb_config(yaml.full_load(f))
    with open(patched_config_path, "w") as f:
        yaml.dump(cfg, f)

    return epoch_dir