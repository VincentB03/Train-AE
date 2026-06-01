r"""Neural network custom layer implementation"""

import math
import equinox as eqx
import jax
from jax import Array
import jax.numpy as jnp

from einops import rearrange

from typing import Union, Sequence, Callable


class ResidualSelfAttention(eqx.Module):
    attention_fn: eqx.nn.MultiheadAttention

    def __init__(self, channels: int, num_heads: int, key: Array = None):
        self.attention_fn = eqx.nn.MultiheadAttention(
            query_size=channels, num_heads=num_heads, key=key
        )

    def __call__(self, x: Array, key: Array = None) -> Array:
        h_ = x.shape[-2]
        w_ = x.shape[-1]

        h = rearrange(x, "... c h w -> ... (h w) c")
        h = self.attention_fn(h, h, h, key=key)
        h = rearrange(h, "... (h w) c-> ... c h w", h=h_, w=w_)

        return x + h


class LayerNorm(eqx.Module):
    r"""Layer Normalization."""

    dim: Union[int, Sequence[int]] = (eqx.field(static=True),)
    eps: float = eqx.field(static=True)

    def __init__(self, dim: Union[int, Sequence[int]], eps: float = 1e-5):
        self.dim = dim
        self.eps = eps

    def __call__(self, x):
        mean = jnp.mean(x, axis=self.dim, keepdims=True)
        var = jnp.var(x, axis=self.dim, keepdims=True)
        var = jnp.maximum(0.0, var)
        inv = jax.lax.rsqrt(var + self.eps)
        return (x - mean) * inv


class SiLU(eqx.Module):
    """SiLU activation function."""

    def __call__(self, x: Array, **kwargs) -> Array:
        return jax.nn.silu(x)


class ChannelAveraging(eqx.Module):
    patch_size: int = eqx.field(static=True)
    group_size: int = eqx.field(static=True)

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int = 1,
    ):
        if isinstance(patch_size, int):
            patch_size = [patch_size] * 2

        self.patch_size = patch_size
        out_factor = math.prod(patch_size)
        assert out_channels % out_factor == 0
        assert in_channels * out_factor % out_channels == 0
        self.group_size = in_channels * out_factor // out_channels

    def __call__(self, x: Array, key: Array = None) -> Array:
        r1, r2 = self.patch_size

        # space to channel
        y = rearrange(x, "... C (H r1) (W r2) -> ... (C r1 r2) H W", r1=r1, r2=r2)
        y = rearrange(y, "(C g) H W-> C g H W", g=self.group_size)

        # channel averaging
        y = y.mean(-3)

        return y


class ChannelDuplicating(eqx.Module):
    patch_size: int = eqx.field(static=True)
    factor: int

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int = 1,
    ):
        if isinstance(patch_size, int):
            patch_size = [patch_size] * 2

        self.patch_size = patch_size
        assert math.prod(self.patch_size) * out_channels % in_channels == 0
        self.factor = math.prod(self.patch_size) * out_channels // in_channels

    def __call__(self, x: Array, key: Array = None) -> Array:
        r1, r2 = self.patch_size

        # channel to space
        y = rearrange(x, "... (C r1 r2) H W -> ... C (H r1) (W r2)", r1=r1, r2=r2)

        # channel duplicating
        y = jnp.concatenate([y for _ in range(self.factor)], axis=-3)

        return y


class Saturate(eqx.Module):
    r"""Saturating function for latent space regularization

    References:
    | Lost in Latent Space: An Empirical Study of Latent Diffusion Models for Physics Emulation (Rozet et al., 2025)
    | https://arxiv.org/abs/2507.02608
    """

    fn: Callable

    def __init__(self, saturation: str = "softclip2"):
        if saturation is None:
            self.fn = lambda x: x
        elif saturation == "softclip":
            self.fn = lambda x: x / (1 + abs(x) / 5)
        elif saturation == "softclip2":
            self.fn = lambda x: x * jax.lax.rsqrt(1 + (x / 5) ** 2)
        else:
            raise NotImplementedError()

    def __call__(self, x: Array) -> Array:
        return self.fn(x)