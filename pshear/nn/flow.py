r"""Latent normalizing flow module"""

import math

import equinox as eqx
from jax import Array
import jax.numpy as jnp

from einops import rearrange

from flowjax.flows import masked_autoregressive_flow, coupling_flow
from flowjax.bijections import RationalQuadraticSpline
from flowjax.distributions import Normal

from typing import Optional, Sequence


def _make_transformer(bijector: str, knots: int, interval: int):
    if bijector == "RQS":
        return RationalQuadraticSpline(knots=knots, interval=interval)
    elif bijector == "affine":
        return None  # let flowjax use its default affine with min scale constraint
    else:
        raise ValueError(f"{bijector} is not yet implemented")


class LatentFlow(eqx.Module):
    r"""Latent Normalizing Flow module"""

    latent_dim: Sequence[int]
    cond_dim: Optional[int]
    flow: eqx.Module

    def __init__(
        self,
        latent_dim: Sequence[int],
        cond_dim: Optional[int],
        bijector: str,
        nn_width: int,
        nn_depth: int,
        knots: int,
        interval: int,
        flow_type: str = "MAF",
        flow_layers: int = 8,
        key: Array = None,
    ):
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim

        transformer = _make_transformer(bijector, knots, interval)
        base_dist = Normal(jnp.zeros(math.prod(latent_dim)))

        if flow_type == "MAF":
            self.flow = masked_autoregressive_flow(
                key,
                base_dist=base_dist,
                transformer=transformer,
                cond_dim=cond_dim,
                nn_width=nn_width,
                nn_depth=nn_depth,
            )
        elif flow_type == "RealNVP":
            self.flow = coupling_flow(
                key,
                base_dist=base_dist,
                transformer=transformer,
                cond_dim=cond_dim,
                flow_layers=flow_layers,
                nn_width=nn_width,
                nn_depth=nn_depth,
                invert=False,
            )
        else:
            raise ValueError(f"flow_type={flow_type} is not implemented")

    @eqx.filter_jit
    def forward(self, *args, **kwargs) -> Array:
        """Apply the flow forward transformation"""
        return self.flow.bijection.transform(*args, **kwargs)

    @eqx.filter_jit
    def sample(self, *args, **kwargs) -> Array:
        return self.flow.sample(*args, **kwargs)

    @eqx.filter_jit
    def log_prob(self, *args, **kwargs) -> Array:
        return self.flow.log_prob(*args, **kwargs)

    @eqx.filter_jit
    def flatten_latent(self, x: Array) -> Array:
        return rearrange(x, "... c h w -> ... (c h w)")

    @eqx.filter_jit
    def unflatten_latent(self, x: Array):
        return rearrange(
            x,
            "... (c h w) -> ... c h w",
            c=self.latent_dim[0],
            h=self.latent_dim[1],
            w=self.latent_dim[2],
        )


def make_latent_flow(
    latent_dim: Sequence[int] = [1, 4, 4],
    cond_dim: Optional[int] = None,
    bijector="RQS",
    nn_width: int = 256,
    nn_depth: int = 8,
    knots: int = 32,
    interval: int = 16,
    flow_type: str = "MAF",
    flow_layers: int = 8,
    key: Array = None,
    **absorb,
):
    flow = LatentFlow(
        latent_dim=latent_dim,
        cond_dim=cond_dim,
        bijector=bijector,
        key=key,
        nn_width=nn_width,
        nn_depth=nn_depth,
        knots=knots,
        interval=interval,
        flow_type=flow_type,
        flow_layers=flow_layers,
    )

    return flow
