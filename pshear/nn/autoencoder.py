r"""Auto-encoder neural network architecture"""

import equinox as eqx

from jax import Array

from .layers import Saturate
from .utils import split

from typing import Callable


class AutoEncoder(eqx.Module):
    r"""Creates an auto-encoder module."""

    saturate: Callable
    encoder: eqx.Module
    decoder: eqx.Module

    def __init__(
        self,
        encoder: eqx.Module,
        decoder: eqx.Module,
        saturation: str = "softclip2",
        *args,
        **kwargs,
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.saturate = Saturate(saturation=saturation)

    def encode(self, x: Array, key: Array = None) -> Array:
        z = self.encoder(x, key)
        z = self.saturate(z)
        return z

    def decode(self, x: Array, key: Array = None) -> Array:
        return self.decoder(x, key)

    def __call__(self, x: Array, key: Array = None) -> Array:
        keys = split(key, 2)
        z = self.encode(x, keys[0])
        y = self.decode(z, keys[1])
        return y, z