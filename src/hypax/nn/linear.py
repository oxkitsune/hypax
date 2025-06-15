# hyperbolic_linear.py
#
# JAX/nnx implementation of a Poincaré fully-connected layer that mirrors
# `hypll.layers.HLinear` (PyTorch) while following the style of nnx.Linear.

from __future__ import annotations
import typing as tp

import jax.numpy as jnp
from flax import nnx


from hypll.manifolds import Manifold
from hypll.tensors import ManifoldTensor
from hypll.utils.layer_utils import (
    check_if_man_dims_match,
    check_if_manifolds_match,
)

Dtype = jnp.dtype


class HLinear(nnx.Module):
    """Hyperbolic (Poincaré) fully-connected layer for JAX/nnx."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        manifold: Manifold,
        *,
        use_bias: bool = True,
        dtype: tp.Optional[Dtype] = None,
        param_dtype: Dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        """
        Args
        ----
        in_features / out_features : Input & output feature sizes.
        manifold                  : Instance of `hypll.manifolds.Manifold`.
        use_bias                  : Attach bias term (default **True**).
        dtype                     : Computation dtype (default: infer).
        param_dtype               : Parameter dtype (default **float32**).
        rngs                      : `nnx.Rngs` container (use `.params()`).
        """
        super().__init__()

        # Store config.
        self.in_features = in_features
        self.out_features = out_features
        self.manifold = manifold
        self.use_bias = use_bias
        self.dtype = dtype
        self.param_dtype = param_dtype

        # ── Parameter initialisation on the manifold ────────────────────────────
        # `construct_dl_parameters` is expected to return the tangent–space
        # weight (z) and an optional bias already placed on the manifold.
        z_key = rngs.params()
        b_key = rngs.params() if use_bias else None
        z_init, bias_init = self.manifold.construct_dl_parameters(
            in_features=in_features,
            out_features=out_features,
            bias=use_bias,
            key_z=z_key,
            key_bias=b_key,
            dtype=param_dtype,
        )

        self.z = nnx.Param(z_init)  # shape: (in_features, out_features)
        self.bias = nnx.Param(bias_init)  # shape: (out_features,) or None

    # --------------------------------------------------------------------- forward
    def __call__(self, x: ManifoldTensor) -> ManifoldTensor:
        """Applies the hyperbolic fully-connected transformation."""
        # Safety checks (same helpers as the PyTorch version).
        check_if_manifolds_match(layer=self, input=x)
        check_if_man_dims_match(layer=self, man_dim=-1, input=x)

        return self.manifold.fully_connected(  # delegates all geometry
            x=x,
            z=self.z.value,
            bias=self.bias.value,
        )

    # ------------------------------------------------------------------ utilities
    def reset_parameters(self, rngs: nnx.Rngs) -> None:
        """Re-initialise parameters (mirrors the PyTorch `reset_parameters`)."""
        z_key = rngs.params()
        b_key = rngs.params() if self.use_bias else None
        z_init, bias_init = self.manifold.construct_dl_parameters(
            in_features=self.in_features,
            out_features=self.out_features,
            bias=self.use_bias,
            key_z=z_key,
            key_bias=b_key,
            dtype=self.param_dtype,
        )
        self.z.value = z_init
        if self.use_bias:
            self.bias.value = bias_init
