# hyperbolic_linear.py
#
# JAX/nnx implementation of a Poincaré fully-connected layer that mirrors
# `hypll.layers.HLinear` (PyTorch) while following the style of nnx.Linear.

from __future__ import annotations
import typing as tp

import jax.numpy as jnp
from flax import nnx

from hypax.array import ManifoldArray
from hypax.manifolds import Manifold

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
        manifold                  : Instance of `hypax.manifolds.Manifold`.
        use_bias                  : Attach bias term (default **True**).
        dtype                     : Computation dtype (default: infer).
        param_dtype               : Parameter dtype (default **float32**).
        rngs                      : `nnx.Rngs` container (use `.params()`).
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.manifold = manifold
        self.use_bias = use_bias
        self.dtype = dtype
        self.param_dtype = param_dtype

        z_key = rngs.params()
        b_key = rngs.params() if use_bias else None
        weights, bias_value = self.manifold.construct_dl_parameters(
            in_features=in_features,
            out_features=out_features,
            bias=use_bias,
            key_z=z_key,
            key_bias=b_key,
            dtype=param_dtype,
        )

        self.weights = nnx.Param(jnp.asarray(weights, dtype=param_dtype))
        self.bias = (
            nnx.Param(jnp.asarray(bias_value, dtype=param_dtype))
            if bias_value is not None
            else None
        )

    def __call__(self, x: ManifoldArray) -> ManifoldArray:
        """Apply the hyperbolic fully connected operation."""
        if not isinstance(x, ManifoldArray):
            raise TypeError(f"Input must be a ManifoldArray, got {type(x)}")
        if x.manifold is not self.manifold:
            raise ValueError("Input manifold does not match layer manifold")
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected last dimension {self.in_features}, got {x.shape[-1]}"
            )

        bias_value = self.bias.value if (self.bias is not None and self.use_bias) else None
        result = self.manifold.fully_connected(
            x=x.array,
            z=self.weights.value,
            bias=bias_value,
            axis=-1,
        )

        return ManifoldArray(data=result, manifold=self.manifold)

    def reset_parameters(self, rngs: nnx.Rngs) -> None:
        """Re-initialize weights/bias."""
        z_key = rngs.params()
        b_key = rngs.params() if self.use_bias else None
        weights, bias_value = self.manifold.construct_dl_parameters(
            in_features=self.in_features,
            out_features=self.out_features,
            bias=self.use_bias,
            key_z=z_key,
            key_bias=b_key,
            dtype=self.param_dtype,
        )
        self.weights.value = jnp.asarray(weights, dtype=self.param_dtype)
        if self.bias is not None:
            self.bias.value = (
                jnp.asarray(bias_value, dtype=self.param_dtype)
                if bias_value is not None
                else None
            )
