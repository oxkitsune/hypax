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
        #
        # Support both PyTorch (hypll) and JAX (hypax) manifolds:
        # - PyTorch version: construct_dl_parameters(in_features, out_features, bias)
        # - JAX version: construct_dl_parameters(..., key_z=..., key_bias=..., dtype=...)
        import inspect
        sig = inspect.signature(self.manifold.construct_dl_parameters)

        z_key = rngs.params()
        b_key = rngs.params() if use_bias else None

        # Check if manifold accepts JAX-style kwargs
        if 'key_z' in sig.parameters:
            # JAX-style manifold (hypax)
            z_init, bias_init = self.manifold.construct_dl_parameters(
                in_features=in_features,
                out_features=out_features,
                bias=use_bias,
                key_z=z_key,
                key_bias=b_key,
                dtype=param_dtype,
            )
            # Convert to JAX arrays if needed
            z_init = jnp.asarray(z_init) if hasattr(z_init, '__array__') else z_init
            if bias_init is not None:
                bias_init = jnp.asarray(bias_init) if hasattr(bias_init, '__array__') else bias_init
        else:
            # PyTorch-style manifold (hypll) - construct then initialize
            z_param, bias_param = self.manifold.construct_dl_parameters(
                in_features=in_features,
                out_features=out_features,
                bias=use_bias,
            )

            # Initialize using manifold's reset_parameters
            self.manifold.reset_parameters(z_param, bias_param)

            # Extract underlying tensors and convert to JAX arrays
            # ManifoldParameter has .tensor attribute for underlying data
            z_tensor = z_param.tensor if hasattr(z_param, 'tensor') else z_param
            z_init = jnp.asarray(z_tensor.detach().cpu().numpy())

            if bias_param is not None:
                bias_tensor = bias_param.tensor if hasattr(bias_param, 'tensor') else bias_param
                bias_init = jnp.asarray(bias_tensor.detach().cpu().numpy()) if bias_tensor is not None else None
            else:
                bias_init = None

        self.z = nnx.Param(z_init)  # shape: (in_features, out_features)
        self.bias = nnx.Param(bias_init)  # shape: (out_features,) or None

    # --------------------------------------------------------------------- forward
    def __call__(self, x: ManifoldTensor) -> ManifoldTensor:
        """Applies the hyperbolic fully-connected transformation."""
        # Safety checks (same helpers as the PyTorch version).
        check_if_manifolds_match(layer=self, input=x)
        check_if_man_dims_match(layer=self, man_dim=-1, input=x)

        # Convert PyTorch ManifoldTensor to JAX arrays for computation
        x_jax = jnp.asarray(x.tensor.detach().cpu().numpy())

        # Perform fully connected operation in JAX
        # Check if we're using a JAX manifold or PyTorch manifold
        import inspect
        sig = inspect.signature(self.manifold.fully_connected)

        # For PyTorch manifolds, we need to use the manifold's fully_connected method
        # For JAX manifolds, we can call it directly
        if hasattr(self.manifold, 'curvature'):
            # Use JAX implementation directly
            from hypax.manifolds.poincare_ball._linalg import poincare_fully_connected
            c_value = self.manifold.curvature.value
            result_jax = poincare_fully_connected(
                x=x_jax,
                z=self.z.value,
                bias=self.bias.value if self.use_bias else None,
                c=c_value,
                axis=x.man_dim,
            )
            # Project to ensure it's on the manifold
            from hypax.manifolds.poincare_ball._diffgeom import project
            result_jax = project(result_jax, c_value, axis=x.man_dim)
        else:
            # Fallback to manifold's method (shouldn't happen in practice)
            result_jax = self.manifold.fully_connected(
                x=x_jax,
                z=self.z.value,
                bias=self.bias.value,
            )

        # Convert result back to PyTorch ManifoldTensor
        import torch
        import numpy as np
        result_torch = torch.from_numpy(np.array(result_jax))

        return ManifoldTensor(
            data=result_torch,
            manifold=self.manifold,
            man_dim=x.man_dim,
        )

    # ------------------------------------------------------------------ utilities
    def reset_parameters(self, rngs: nnx.Rngs) -> None:
        """Re-initialise parameters (mirrors the PyTorch `reset_parameters`)."""
        import inspect
        sig = inspect.signature(self.manifold.construct_dl_parameters)

        z_key = rngs.params()
        b_key = rngs.params() if self.use_bias else None

        # Check if manifold accepts JAX-style kwargs
        if 'key_z' in sig.parameters:
            # JAX-style manifold (hypax)
            z_init, bias_init = self.manifold.construct_dl_parameters(
                in_features=self.in_features,
                out_features=self.out_features,
                bias=self.use_bias,
                key_z=z_key,
                key_bias=b_key,
                dtype=self.param_dtype,
            )
            z_init = jnp.asarray(z_init) if hasattr(z_init, '__array__') else z_init
            if bias_init is not None:
                bias_init = jnp.asarray(bias_init) if hasattr(bias_init, '__array__') else bias_init
        else:
            # PyTorch-style manifold (hypll)
            z_param, bias_param = self.manifold.construct_dl_parameters(
                in_features=self.in_features,
                out_features=self.out_features,
                bias=self.use_bias,
            )

            # Initialize using manifold's reset_parameters
            self.manifold.reset_parameters(z_param, bias_param)

            # Extract underlying tensors and convert to JAX arrays
            z_tensor = z_param.tensor if hasattr(z_param, 'tensor') else z_param
            z_init = jnp.asarray(z_tensor.detach().cpu().numpy())

            if bias_param is not None:
                bias_tensor = bias_param.tensor if hasattr(bias_param, 'tensor') else bias_param
                bias_init = jnp.asarray(bias_tensor.detach().cpu().numpy()) if bias_tensor is not None else None
            else:
                bias_init = None

        self.z.value = z_init
        if self.use_bias:
            self.bias.value = bias_init
