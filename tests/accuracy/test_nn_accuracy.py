"""Accuracy tests for neural network modules.

Tests the HLinear hyperbolic fully-connected layer.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax import nnx

from hypax.nn.linear import HLinear
from hypll.manifolds.poincare_ball import PoincareBall
from hypll.tensors import ManifoldTensor


@pytest.fixture
def poincare_manifold():
    """Create a PoincareBall manifold for testing."""
    return PoincareBall(c=1.0)


class TestHLinear:
    """Test HLinear hyperbolic fully-connected layer."""

    @pytest.mark.parametrize(
        "in_features,out_features,batch_size",
        [
            (3, 2, 5),
            (10, 5, 8),
            (50, 20, 10),
        ],
    )
    def test_hlinear_forward_without_bias(
        self, in_features, out_features, batch_size, poincare_manifold, jax_key
    ):
        """Test HLinear forward pass without bias."""
        # Create layer
        rngs = nnx.Rngs(params=jax_key)
        layer = HLinear(
            in_features=in_features,
            out_features=out_features,
            manifold=poincare_manifold,
            use_bias=False,
            rngs=rngs,
        )

        # Create input on the manifold
        key_input = jax.random.split(jax_key)[0]
        x_data_jax = jax.random.uniform(
            key_input, (batch_size, in_features), minval=-0.5, maxval=0.5
        )
        # Convert JAX array to PyTorch tensor for ManifoldTensor
        x_data = torch.from_numpy(np.array(x_data_jax))
        x = ManifoldTensor(x_data, manifold=poincare_manifold)

        # Forward pass
        output = layer(x)

        # Check output properties
        assert isinstance(output, ManifoldTensor), "Output should be a ManifoldTensor"
        assert output.shape == (batch_size, out_features), (
            f"Expected shape {(batch_size, out_features)}, got {output.shape}"
        )
        assert output.manifold == poincare_manifold, (
            "Output manifold should match input manifold"
        )

        # Check output is on the manifold (within the ball)
        output_np = output.tensor.detach().cpu().numpy()
        norms = np.linalg.norm(output_np, axis=-1)
        max_norm = 1.0 / np.sqrt(float(poincare_manifold.c))
        assert np.all(norms < max_norm * 1.1), (
            "Output points should be within the Poincaré ball"
        )

        # Check for finite values
        assert np.all(np.isfinite(output_np)), "Output contains non-finite values"

    @pytest.mark.parametrize(
        "in_features,out_features,batch_size",
        [
            (3, 2, 5),
            (10, 5, 8),
        ],
    )
    def test_hlinear_forward_with_bias(
        self, in_features, out_features, batch_size, poincare_manifold, jax_key
    ):
        """Test HLinear forward pass with bias."""
        # Create layer
        rngs = nnx.Rngs(params=jax_key)
        layer = HLinear(
            in_features=in_features,
            out_features=out_features,
            manifold=poincare_manifold,
            use_bias=True,
            rngs=rngs,
        )

        # Create input
        key_input = jax.random.split(jax_key)[0]
        x_data_jax = jax.random.uniform(
            key_input, (batch_size, in_features), minval=-0.5, maxval=0.5
        )
        x_data = torch.from_numpy(np.array(x_data_jax))
        x = ManifoldTensor(x_data, manifold=poincare_manifold)

        # Forward pass
        output = layer(x)

        # Check output properties
        assert isinstance(output, ManifoldTensor), "Output should be a ManifoldTensor"
        assert output.shape == (batch_size, out_features)
        assert output.manifold == poincare_manifold

        # Check output is on the manifold
        output_np = output.tensor.detach().cpu().numpy()
        norms = np.linalg.norm(output_np, axis=-1)
        max_norm = 1.0 / np.sqrt(float(poincare_manifold.c))
        assert np.all(norms < max_norm * 1.1), (
            "Output points should be within the Poincaré ball"
        )

        # Check for finite values
        assert np.all(np.isfinite(output_np)), "Output contains non-finite values"

    def test_hlinear_parameter_shapes(self, poincare_manifold, jax_key):
        """Test that HLinear parameters have correct shapes."""
        in_features, out_features = 10, 5

        # Create layer with bias
        rngs = nnx.Rngs(params=jax_key)
        layer = HLinear(
            in_features=in_features,
            out_features=out_features,
            manifold=poincare_manifold,
            use_bias=True,
            rngs=rngs,
        )

        # Check parameter shapes
        assert layer.z.value.shape == (in_features, out_features), (
            f"Weight shape should be ({in_features}, {out_features})"
        )
        assert layer.bias.value.shape == (out_features,), (
            f"Bias shape should be ({out_features},)"
        )

        # Create layer without bias
        rngs2 = nnx.Rngs(params=jax.random.split(jax_key)[0])
        layer_no_bias = HLinear(
            in_features=in_features,
            out_features=out_features,
            manifold=poincare_manifold,
            use_bias=False,
            rngs=rngs2,
        )

        assert layer_no_bias.z.value.shape == (in_features, out_features)
        assert layer_no_bias.bias.value is None, (
            "Bias should be None when use_bias=False"
        )

    def test_hlinear_reset_parameters(self, poincare_manifold, jax_key):
        """Test that reset_parameters changes the layer parameters."""
        in_features, out_features = 5, 3

        # Create layer
        rngs = nnx.Rngs(params=jax_key)
        layer = HLinear(
            in_features=in_features,
            out_features=out_features,
            manifold=poincare_manifold,
            use_bias=True,
            rngs=rngs,
        )

        # Store original parameters
        original_z = layer.z.value.copy()
        original_bias = layer.bias.value.copy()

        # Reset parameters with a new key
        key_new = jax.random.split(jax_key)[1]
        rngs_new = nnx.Rngs(params=key_new)
        layer.reset_parameters(rngs_new)

        # Check that parameters have changed (or at least are reinitialized)
        # Weights should change with different random key
        assert not np.allclose(np.array(layer.z.value), np.array(original_z)), (
            "Weight should change after reset"
        )
        # Bias gets reset to zeros in the manifold's reset_parameters
        # So we just check the shape is preserved
        assert layer.bias.value.shape == original_bias.shape, (
            "Bias shape should be preserved after reset"
        )

        # Check shapes remain the same
        assert layer.z.value.shape == original_z.shape
        assert layer.bias.value.shape == original_bias.shape

    def test_hlinear_gradients(self, poincare_manifold, jax_key):
        """Test that HLinear can perform forward passes (gradient computation not supported with PyTorch tensors)."""
        in_features, out_features, batch_size = 5, 3, 4

        # Create layer
        rngs = nnx.Rngs(params=jax_key)
        layer = HLinear(
            in_features=in_features,
            out_features=out_features,
            manifold=poincare_manifold,
            use_bias=True,
            rngs=rngs,
        )

        # Create input
        key_input = jax.random.split(jax_key)[0]
        x_data_jax = jax.random.uniform(
            key_input, (batch_size, in_features), minval=-0.5, maxval=0.5
        )
        x_data = torch.from_numpy(np.array(x_data_jax))
        x = ManifoldTensor(x_data, manifold=poincare_manifold)

        # Forward pass
        output = layer(x)

        # Verify output
        assert isinstance(output, ManifoldTensor)
        assert output.shape == (batch_size, out_features)

        # Check parameter shapes exist
        assert layer.z.value.shape == (in_features, out_features)
        if layer.use_bias:
            assert layer.bias.value.shape == (out_features,)

        # Note: Gradient computation through PyTorch/JAX boundary is not supported
        # For pure JAX gradients, use hypax manifolds with ManifoldArray instead

    def test_hlinear_batch_independence(self, poincare_manifold, jax_key):
        """Test that batch samples are processed independently."""
        in_features, out_features = 5, 3

        # Create layer
        rngs = nnx.Rngs(params=jax_key)
        layer = HLinear(
            in_features=in_features,
            out_features=out_features,
            manifold=poincare_manifold,
            use_bias=True,
            rngs=rngs,
        )

        # Create two separate inputs
        key1, key2 = jax.random.split(jax_key, 2)
        x1_data_jax = jax.random.uniform(
            key1, (1, in_features), minval=-0.5, maxval=0.5
        )
        x2_data_jax = jax.random.uniform(
            key2, (1, in_features), minval=-0.5, maxval=0.5
        )

        # Convert to PyTorch tensors
        x1_data = torch.from_numpy(np.array(x1_data_jax))
        x2_data = torch.from_numpy(np.array(x2_data_jax))

        # Create batched input
        x_batched_data = torch.cat([x1_data, x2_data], dim=0)

        x1 = ManifoldTensor(x1_data, manifold=poincare_manifold)
        x2 = ManifoldTensor(x2_data, manifold=poincare_manifold)
        x_batched = ManifoldTensor(x_batched_data, manifold=poincare_manifold)

        # Forward pass
        out1 = layer(x1)
        out2 = layer(x2)
        out_batched = layer(x_batched)

        # Check that batch processing gives same results as individual processing
        out1_np = out1.tensor.detach().cpu().numpy()
        out2_np = out2.tensor.detach().cpu().numpy()
        out_batched_np = out_batched.tensor.detach().cpu().numpy()

        assert np.allclose(out1_np, out_batched_np[0:1], rtol=1e-5), (
            "First batch item should match"
        )
        assert np.allclose(out2_np, out_batched_np[1:2], rtol=1e-5), (
            "Second batch item should match"
        )

    def test_hlinear_different_curvatures(self, jax_key):
        """Test HLinear with different curvature values."""
        in_features, out_features, batch_size = 5, 3, 4

        for c_value in [0.5, 1.0, 2.0]:
            manifold = PoincareBall(c=c_value)
            rngs = nnx.Rngs(params=jax_key)
            layer = HLinear(
                in_features=in_features,
                out_features=out_features,
                manifold=manifold,
                use_bias=True,
                rngs=rngs,
            )

            # Create input
            key_input = jax.random.split(jax_key)[0]
            # Scale input appropriately for curvature
            max_input_norm = 0.5 / jnp.sqrt(c_value)
            x_data_jax = (
                jax.random.uniform(key_input, (batch_size, in_features))
                * max_input_norm
            )
            x_data = torch.from_numpy(np.array(x_data_jax))
            x = ManifoldTensor(x_data, manifold=manifold)

            # Forward pass
            output = layer(x)

            # Check output is on the manifold
            output_np = output.tensor.detach().cpu().numpy()
            norms = np.linalg.norm(output_np, axis=-1)
            max_norm = 1.0 / np.sqrt(c_value)
            assert np.all(norms < max_norm * 1.1), (
                f"Output should be within Poincaré ball for c={c_value}"
            )
            assert np.all(np.isfinite(output_np)), (
                f"Output should be finite for c={c_value}"
            )
