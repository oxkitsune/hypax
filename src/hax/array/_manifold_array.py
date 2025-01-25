import jax
import jax.numpy as jnp

from hax.manifolds import Manifold


# perhaps we should *extend* jax.Array, and make a "DeviceManifoldArray" etc?
# however, this will make hax more complex
class ManifoldArray:
    def __init__(self, data: jax.Array, manifold: Manifold):
        self.array = data
        self.manifold = manifold

    @property
    def ndim(self):
        return self.array.ndim

    @property
    def shape(self):
        return self.array.shape
