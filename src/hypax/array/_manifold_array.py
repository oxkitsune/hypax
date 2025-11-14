import jax
import jax.numpy as jnp

from hypax.manifolds import Manifold


# perhaps we should *extend* jax.Array, and make a "DeviceManifoldArray" etc?
# however, this will make hypax more complex
class ManifoldArray:
    def __init__(self, data: jax.Array, manifold: Manifold):
        self.array = data
        self.manifold = manifold

    @property
    def ndim(self):
        return self.array.ndim

    def dim(self) -> int:
        """PyTorch-style alias used by some shared helpers."""
        return self.array.ndim

    @property
    def shape(self):
        return self.array.shape
