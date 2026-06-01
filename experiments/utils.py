import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from einops import rearrange

if "SCRATCH" in os.environ:
    PATH = Path(os.environ["SCRATCH"]) / "pshear/cosmos"
else:
    PATH = Path(".")
PATH.mkdir(parents=True, exist_ok=True)

def plot_ae_residuals(batch, y, path=None):
    x = np.concatenate([batch["obs"], y], 1)
    x = np.concatenate([x, batch["obs"] - y], 1)[:12]
    x /= x.max((-1, -2), keepdims=True)
    image = rearrange(x, "(b1 b2) c h w -> (b1 h) (b2 c w)", b1=6, b2=2)
    if path is not None:
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.axis("off")
        plt.savefig(path)
        plt.close()
    cmap = cm.get_cmap("viridis")
    normed_image = (image - np.min(image)) / (np.max(image) - np.min(image))
    rgba_image = cmap(normed_image)
    rgb_image = (rgba_image[..., :3] * 255).astype(np.uint8)
    return rgb_image