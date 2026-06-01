r"""Neural network building blocks"""

import math
import equinox as eqx
import equinox.nn as nn
import jax
from jax import Array

from einops import rearrange

from typing import Optional, Sequence, Dict

from .layers import (
    SiLU,
    LayerNorm,
    ChannelAveraging,
    ChannelDuplicating,
    ResidualSelfAttention,
)

from .utils import split


class ResBlock(eqx.Module):
    r"""Creates a residua block"""

    norm: eqx.Module
    block: eqx.Module
    residual_attention: eqx.Module

    def __init__(
        self,
        channels: int,
        norm: str = "layer",  # or "group"
        groups: int = 16,
        dropout: Optional[float] = None,
        attention_heads: Optional[int] = None,
        key: Array = None,
        **kwargs,
    ):
        keys = jax.random.split(key, 3)

        if norm == "layer":
            self.norm = LayerNorm(dim=-3)  # across channels
        elif norm == "group":
            self.norm = nn.GroupNorm(groups=min(groups, channels), channels=channels)
        else:
            raise NotImplementedError()

        if attention_heads is None:
            self.residual_attention = nn.Identity()
        else:
            self.residual_attention = ResidualSelfAttention(
                channels=channels, num_heads=attention_heads, key=keys[0]
            )

        self.block = nn.Sequential(
            [
                nn.Conv2d(channels, channels, key=keys[1], **kwargs),
                SiLU(),
                nn.Identity() if dropout is None else nn.Dropout(dropout),
                nn.Conv2d(channels, channels, key=keys[2], **kwargs),
            ]
        )

    def __call__(self, x: Array, key: Array = None) -> Array:
        keys = split(key, 2)

        h = self.norm(x)
        h = self.residual_attention(h, key=keys[0])
        h = self.block(h, key=keys[1])

        return x + h


class Sum(eqx.Module):
    block1: eqx.Module
    block2: eqx.Module

    def __init__(self, block1: eqx.Module, block2: eqx.Module):
        self.block1 = block1
        self.block2 = block2

    def __call__(self, x: Array, key: Array = None) -> Array:
        keys = split(key, 2)
        return self.block1(x, key=keys[0]) + self.block2(x, key=keys[1])


class DownBlock(eqx.Module):
    r"""Creates a down sampling block module."""

    down_layer: eqx.Module
    patch_size: int = eqx.field(static=True)

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        patch_size: int = 1,
        key: Array = None,
        **kwargs,
    ):
        if isinstance(stride, int):
            stride = [stride] * 2

        if isinstance(patch_size, int):
            patch_size = [patch_size] * 2

        self.patch_size = patch_size
        out_factor = math.prod(patch_size)
        assert out_channels % out_factor == 0

        self.down_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels // out_factor,
            stride=stride,
            key=key,
            **kwargs,
        )

    def __call__(self, x: Array, key: Array = None) -> Array:
        r1, r2 = self.patch_size
        y = self.down_layer(x, key=key)
        y = rearrange(y, "... C (H r1) (W r2) -> ... (C r1 r2) H W", r1=r1, r2=r2)

        return y


class UpBlock(eqx.Module):
    r"""Creates a up sampling block module."""

    up_layer: eqx.Module
    patch_size: int = eqx.field(static=True)

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int = 1,
        key: Array = None,
        **kwargs,
    ):
        if isinstance(patch_size, int):
            patch_size = [patch_size] * 2

        self.patch_size = patch_size
        out_factor = math.prod(patch_size)

        self.up_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels * out_factor,
            key=key,
            **kwargs,
        )

    def __call__(self, x: Array, key: Array = None) -> Array:
        r1, r2 = self.patch_size
        y = self.up_layer(x, key=key)
        y = rearrange(y, "... (C r1 r2) H W -> ... C (H r1) (W r2)", r1=r1, r2=r2)

        return y


class Encoder(eqx.Module):
    r"""Creates a Deep Compressor encoder module.

    Assuming a patch size r
    - Space to Channel (C,rR -> 2rC, R).
    - Channel Averaging (2rC, R -> rC, R)

    References:
    | Deep Compression Autoencoder for Efficient High-Resolution Diffusion Models (Chen et al., 2024)
    | https://arxiv.org/abs/2410.10733v1
    """

    down: Sequence[Sequence[eqx.Module]]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hid_channels: Sequence[int] = (64, 128, 256),
        hid_blocks: Sequence[int] = (3, 3, 3),
        attention_heads: Dict[int, int] = {},
        dropout: float = None,
        norm: str = "layer",
        patch_size: int = 1,
        stride: int = 2,
        key: Array = None,
        **kwargs,
    ):
        assert len(hid_channels) == len(hid_blocks)

        keys = jax.random.split(key, 5)
        self.down = []

        self.down.append(
            [
                DownBlock(
                    in_channels=in_channels,
                    out_channels=hid_channels[0],
                    patch_size=patch_size,
                    key=keys[0],
                    **kwargs,
                )
            ]
        )

        for i in range(len(hid_channels) - 1):
            down_block = []
            sub_keys = jax.random.split(keys[1], hid_blocks[i])
            for j in range(hid_blocks[i]):
                down_block.append(
                    ResBlock(
                        channels=hid_channels[i],
                        norm=norm,
                        dropout=dropout,
                        attention_heads=attention_heads.get(i, None),
                        key=sub_keys[j],
                        **kwargs,
                    )
                )

            down_block.append(
                Sum(
                    DownBlock(
                        in_channels=hid_channels[i],
                        out_channels=hid_channels[i + 1],
                        patch_size=1,
                        stride=stride,
                        key=keys[2],
                        **kwargs,
                    ),
                    ChannelAveraging(
                        in_channels=hid_channels[i],
                        out_channels=hid_channels[i + 1],
                        patch_size=stride * patch_size,
                    ),
                )
            )

            self.down.append(down_block)

        down_block = []
        sub_keys = jax.random.split(keys[3], hid_blocks[-1])
        for j in range(hid_blocks[-1]):
            down_block.append(
                ResBlock(
                    channels=hid_channels[-1],
                    norm=norm,
                    dropout=dropout,
                    attention_heads=attention_heads.get(len(hid_channels) - 1, None),
                    key=sub_keys[j],
                    **kwargs,
                )
            )

        down_block.append(
            DownBlock(
                in_channels=hid_channels[-1],
                out_channels=out_channels,
                key=keys[4],
                **kwargs,
            )
        )

        self.down.append(down_block)

    def __call__(self, x: Array, key: Array = None) -> Array:
        keys = split(key, len(self.down))

        for i, blocks in enumerate(self.down):
            sub_keys = split(keys[i], len(blocks))

            for j, block in enumerate(blocks):
                x = block(x, sub_keys[j])

        return x


class Decoder(eqx.Module):
    r"""Creates a Deep Compressor decoder module.

    References:
    | Deep Compression Autoencoder for Efficient High-Resolution Diffusion Models (Chen et al., 2024)
    | https://arxiv.org/abs/2410.10733v1
    """

    up: Sequence[Sequence[eqx.Module]]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hid_channels: Sequence[int] = (64, 128, 256),
        hid_blocks: Sequence[int] = (3, 3, 3),
        attention_heads: Dict[int, int] = {},
        dropout: float = None,
        norm: str = "layer",
        patch_size: int = 1,
        stride: int = 2,
        key: Array = None,
        **kwargs,
    ):
        assert len(hid_channels) == len(hid_blocks)

        if isinstance(patch_size, int):
            patch_size = [patch_size] * 2

        if isinstance(stride, int):
            stride = [stride] * 2

        keys = jax.random.split(key, 5)
        self.up = []

        self.up.append(
            [
                UpBlock(
                    in_channels=in_channels,
                    out_channels=hid_channels[-1],
                    patch_size=patch_size,
                    key=keys[0],
                    **kwargs,
                )
            ]
        )

        num_blocks = len(hid_channels)
        for i in range(num_blocks - 1):
            up_block = []
            sub_keys = jax.random.split(keys[1], hid_blocks[i])
            for j in range(hid_blocks[i]):
                up_block.append(
                    ResBlock(
                        channels=hid_channels[num_blocks - 1 - i],
                        norm=norm,
                        attention_heads=attention_heads.get(i, None),
                        dropout=dropout,
                        key=sub_keys[j],
                        **kwargs,
                    )
                )

            # channel_factor =
            up_block.append(
                Sum(
                    UpBlock(
                        in_channels=hid_channels[num_blocks - 1 - i],
                        out_channels=hid_channels[num_blocks - 1 - (i + 1)],
                        patch_size=stride,
                        key=keys[2],
                        **kwargs,
                    ),
                    ChannelDuplicating(
                        in_channels=hid_channels[num_blocks - 1 - i],
                        out_channels=hid_channels[num_blocks - 1 - (i + 1)],
                        patch_size=stride,
                    ),
                )
            )

            self.up.append(up_block)

        up_block = []
        sub_keys = jax.random.split(keys[3], hid_blocks[0])
        for j in range(hid_blocks[0]):
            up_block.append(
                ResBlock(
                    channels=hid_channels[0],
                    norm=norm,
                    attention_heads=attention_heads.get(0, None),
                    dropout=dropout,
                    key=sub_keys[j],
                    **kwargs,
                )
            )

        up_block.append(
            UpBlock(
                in_channels=hid_channels[0],
                out_channels=out_channels,
                key=keys[4],
                **kwargs,
            )
        )

        self.up.append(up_block)

    def __call__(self, x: Array, key: Array = None) -> Array:
        keys = split(key, len(self.up))

        for i, blocks in enumerate(self.up):
            sub_keys = split(keys[i], len(blocks))

            for j, block in enumerate(blocks):
                x = block(x, sub_keys[j])

        return x