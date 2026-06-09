import equinox as eqx
import jax
from jax import Array
from jax.nn import softplus
import jax.numpy as jnp

import jax_galsim as jgalsim
from jax_galsim import Image, InterpolatedImage, Convolve

from .nn.autoencoder import AutoEncoder
from .nn.blocks import Encoder, Decoder
from .nn.utils import split

from typing import Sequence, Union, Optional, Dict
from functools import partial


def convolve_galsim(x, psf, nx, ny, scale):
    gsparams = jgalsim.GSParams(minimum_fft_size=nx, maximum_fft_size=nx)
    x = InterpolatedImage(Image(x[0], scale=scale))
    psf = InterpolatedImage(Image(psf[0], scale=scale))
    convolved = Convolve(x, psf).withGSParams(gsparams)
    array2d =convolved.drawImage(nx=nx, ny=ny, scale=scale, method="no_pixel").array
    return jnp.expand_dims(array2d, 0)


class GalaxyAutoEncoder(AutoEncoder):
    use_jax_galsim: bool = eqx.field(static=True, default=True)
    minimum_fft_size: int = eqx.field(static=True, default=128)
    nx: int = eqx.field(static=True, default=128)
    ny: int = eqx.field(static=True, default=128)
    scale: float = eqx.field(static=True, default=0.03)

    def __init__(self, encoder, decoder, use_jax_galsim, minimum_fft_size, nx, ny, scale, saturation="softclip2", **kwargs):
        
        super().__init__(encoder=encoder, decoder=decoder, saturation=saturation, **kwargs)
        
        self.use_jax_galsim = use_jax_galsim
        self.minimum_fft_size = minimum_fft_size
        self.nx = nx
        self.ny = ny
        self.scale = scale

    def decode(self, x, key=None):
        y = self.decoder(x, key)
        y = softplus(y)
        return y

    def convolve(self, x, psf):
        return convolve_galsim(x, psf, self.nx, self.ny, self.scale)

    @eqx.filter_jit
    def __call__(self, x, psf, key=None):
        keys = split(key, 2)
        z = self.encode(x, keys[0])
        g = self.decode(z, keys[1])
        y = self.convolve(g, psf)
        return y, g, z


class GalaxyAutoEncoderLoss(eqx.Module):
    losses: Sequence[str]
    weights: Sequence[str]

    def __init__(self, losses=["mse"], weights=[1.0]):
        assert len(losses) == len(weights)
        self.losses = list(losses)
        self.weights = list(weights)

    @eqx.filter_jit
    def __call__(self, autoencoder, x, psf, rms, mask, key=None, activate=0.0):
        y, g, z = autoencoder(x, psf, key)
        ell = 0
        for name, weight in zip(self.losses, self.weights):
            if name == "mse":
                loss = ((x - y) ** 2).mean()
            elif name == "mae":
                loss = jnp.abs(x - y).mean()
            elif name == "tv":
                grad1 = g[:, :, 1:] - g[:, :, :-1]
                grad2 = g[:, 1:, :] - g[:, :-1, :]
                loss = activate * (jnp.abs(grad1).sum() + jnp.abs(grad2).sum())
            elif name == "likelihood":
                eps = 1e-8 # prevent division by zero
                sigma       = jnp.clip(rms, a_min=eps) 
                variance = sigma**2 
                sq_err = ((x - y) ** 2) / (2 * variance)
                masked_sq_err = sq_err * mask #use of the mask to consider only valid pixels in the loss computation
                loss = masked_sq_err.sum() / (mask.sum() + eps)
            else:
                raise ValueError(f"Loss {name} is not implemented.")
            ell += loss * weight
        return ell


def make_galaxy_autoencoder(
    use_jax_galsim=True, minimum_fft_size=128,
    nx=128, ny=128, scale=0.03,
    in_channels=1, latent_channels=16,
    kernel_size=3, hid_channels=(), hid_blocks=(),
    attention_heads={}, patch_size=1, stride=2,
    dropout=None, key=None, saturation="softclip2",
    **absorb,
):
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size] * 2
    kwargs = dict(kernel_size=kernel_size, padding=[k // 2 for k in kernel_size])
    keys = jax.random.split(key, 2)

    encoder = Encoder(
        in_channels=in_channels, out_channels=latent_channels,
        hid_channels=hid_channels, hid_blocks=hid_blocks,
        attention_heads=attention_heads, patch_size=patch_size,
        stride=stride, dropout=dropout, key=keys[0], **kwargs,
    )
    decoder = Decoder(
        in_channels=latent_channels, out_channels=in_channels,
        hid_channels=hid_channels, hid_blocks=hid_blocks,
        attention_heads=attention_heads, patch_size=patch_size,
        stride=stride, dropout=dropout, key=keys[1], **kwargs,
    )
    return GalaxyAutoEncoder(
        encoder=encoder, decoder=decoder, saturation=saturation,
        nx=nx, ny=ny, scale=scale,
        use_jax_galsim=use_jax_galsim, minimum_fft_size=minimum_fft_size,
    )