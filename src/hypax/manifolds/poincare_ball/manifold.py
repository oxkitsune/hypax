# manifold.py
#
# JAX implementation of the Poincaré ball model of hyperbolic space

from __future__ import annotations
import typing as tp

import jax
import jax.numpy as jnp
from flax import nnx

from hypax.manifolds._base import Manifold
from hypax.manifolds.poincare_ball._linalg import poincare_fully_connected
from hypax.manifolds.poincare_ball._diffgeom import (
    expmap0,
    expmap,
    logmap0,
    logmap,
    project,
    mobius_add,
    dist,
)


class PoincareBall(Manifold):
    """Poincaré ball model of hyperbolic space.

    The Poincaré ball is a conformal model of hyperbolic geometry. Points on
    the manifold are represented as vectors x with ||x|| < 1/√c, where c is
    the curvature.

    Attributes:
        curvature: Curvature of the manifold (scalar > 0). Higher curvature means
                   more "curved" space with a smaller radius.

    Example:
        >>> manifold = PoincareBall(c=1.0)
        >>> # Points must satisfy ||x|| < 1/√c
        >>> x = jnp.array([[0.1, 0.2], [0.3, -0.1]])
        >>> # Use manifold operations like expmap, logmap, etc.
    """

    def __init__(self, c: float | jax.Array = 1.0):
        """Initialize PoincareBall manifold.

        Args:
            c: Curvature parameter (default: 1.0). Must be positive.
               Can be a scalar float or JAX array.
        """
        super().__init__()
        c_array = jnp.asarray(c) if not isinstance(c, jax.Array) else c

        if jnp.any(c_array <= 0):
            raise ValueError(f"Curvature must be positive, got {c}")

        self.curvature = nnx.Param(c_array)

    def expmap(
        self, v: jax.Array, x: jax.Array | None = None, axis: int = -1
    ) -> jax.Array:
        """Exponential map: map from tangent space to manifold.

        Args:
            v: Tangent vector
            x: Base point on manifold (if None, uses origin)
            axis: Manifold dimension axis

        Returns:
            Point on manifold
        """
        if x is None:
            return expmap0(v, self.curvature.value, axis=axis)
        else:
            return expmap(x, v, self.curvature.value, axis=axis)

    def logmap(
        self, y: jax.Array, x: jax.Array | None = None, axis: int = -1
    ) -> jax.Array:
        """Logarithmic map: map from manifold to tangent space.

        Args:
            y: Point on manifold
            x: Base point on manifold (if None, uses origin)
            axis: Manifold dimension axis

        Returns:
            Tangent vector at base point
        """
        if x is None:
            return logmap0(y, self.curvature.value, axis=axis)
        else:
            return logmap(x, y, self.curvature.value, axis=axis)

    def project(self, x: jax.Array, axis: int = -1, eps: float = -1.0) -> jax.Array:
        """Project point to be within the Poincaré ball.

        Args:
            x: Point (possibly outside ball)
            axis: Manifold dimension axis
            eps: Epsilon for numerical stability (if < 0, uses default)

        Returns:
            Point projected to be within ball
        """
        return project(x, self.curvature.value, axis=axis, eps=eps)

    def mobius_add(self, x: jax.Array, y: jax.Array, axis: int = -1) -> jax.Array:
        """Möbius addition in the Poincaré ball.

        Args:
            x: First point on manifold
            y: Second point on manifold
            axis: Manifold dimension axis

        Returns:
            Result of Möbius addition
        """
        return mobius_add(x, y, self.curvature.value, axis=axis)

    def dist(
        self, x: jax.Array, y: jax.Array, axis: int = -1, keepdims: bool = False
    ) -> jax.Array:
        """Hyperbolic distance between points.

        Args:
            x: First point on manifold
            y: Second point on manifold
            axis: Manifold dimension axis
            keepdims: Whether to keep dimensions

        Returns:
            Hyperbolic distance
        """
        return dist(x, y, self.curvature.value, axis=axis, keepdims=keepdims)

    def construct_dl_parameters(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        key_z: jax.Array | None = None,
        key_bias: jax.Array | None = None,
        dtype: jnp.dtype = jnp.float32,
    ) -> tp.Tuple[jax.Array, jax.Array | None]:
        """Construct and initialize parameters for deep learning layers.

        This method initializes weights and biases for hyperbolic neural network
        layers following the initialization strategy from HNN++:
        - For square/tall matrices (in_features <= out_features): identity init
        - For wide matrices (in_features > out_features): normal init

        Args:
            in_features: Number of input features
            out_features: Number of output features
            bias: Whether to create bias parameters
            key_z: JAX random key for weight initialization
            key_bias: JAX random key for bias initialization
            dtype: Data type for parameters

        Returns:
            Tuple of (weights, bias) where:
            - weights: Array of shape (in_features, out_features)
            - bias: Array of shape (out_features,) if bias=True, else None

        References:
            Chen et al. "Fully Hyperbolic Neural Networks" (HNN++), ACL 2022
        """
        if key_z is None:
            key_z = jax.random.PRNGKey(0)

        # Initialize weights using identity or normal initialization
        if in_features <= out_features:
            # Identity initialization for square or tall matrices
            weights = 0.5 * jnp.eye(in_features, out_features, dtype=dtype)
        else:
            # HNN++ initialization for wide matrices
            std = (2 * in_features * out_features) ** -0.5
            weights = (
                jax.random.normal(key_z, shape=(in_features, out_features), dtype=dtype)
                * std
            )

        # Initialize bias to zeros if needed
        if bias:
            bias_value = jnp.zeros(out_features, dtype=dtype)
        else:
            bias_value = None

        return weights, bias_value

    def fully_connected(
        self,
        x: jax.Array,
        z: jax.Array,
        bias: jax.Array | None,
        axis: int = -1,
    ) -> jax.Array:
        """Hyperbolic fully connected layer operation.

        Applies the fully connected transformation in hyperbolic space using
        hyperplane distances and projections.

        Args:
            x: Input array on the manifold
            z: Weight matrix (tangent space)
            bias: Bias vector (if None, no bias)
            axis: Manifold dimension axis

        Returns:
            Output array on the manifold

        References:
            Chen et al. "Fully Hyperbolic Neural Networks" (HNN++), ACL 2022
        """
        result = poincare_fully_connected(
            x=x, z=z, bias=bias, c=self.curvature.value, axis=axis
        )
        # Project result to ensure it stays in the ball
        return self.project(result, axis=axis)
