from abc import ABC
from flax import nnx


class Manifold(nnx.Module, ABC):
    def expmap(self, v): ...
