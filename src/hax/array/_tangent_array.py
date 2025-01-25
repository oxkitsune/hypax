import jax
import jax.numpy as jnp


class TangentArray:
    def __init__(self, data: jax.Array):
        self.array = data
