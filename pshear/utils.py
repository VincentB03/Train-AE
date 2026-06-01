import yaml
import equinox as eqx
from .galaxy import make_galaxy_autoencoder
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