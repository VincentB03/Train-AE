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
    x = np.concatenate([batch["sci_subtracted"], y], 1)
    x = np.concatenate([x, batch["sci_subtracted"] - y], 1)[:8]
    x /= x.max((-1, -2), keepdims=True)
    image = rearrange(x, "(b1 b2) c h w -> (b1 h) (b2 c w)", b1=4, b2=2)
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

def plot_residual_histogram(batch, y, path=None):
    residuals = (batch["sci_subtracted"] - y).flatten()
    valid_residuals = residuals[np.isfinite(residuals)]
    fig, ax = plt.subplots(figsize=(8, 6))
    vmin, vmax = np.percentile(valid_residuals, [1, 99])
    ax.hist(valid_residuals, bins=100, range=(vmin, vmax), color='purple', alpha=0.7, density=True)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label="ZZero (Null Error)")
    ax.set_title("Global distribution of residuals (Observation - Prediction)")
    ax.set_xlabel("Reconstruction error (Observation - Prediction)")
    ax.set_ylabel("Pixel density")
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    if path is not None:
        fig.savefig(path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        return path