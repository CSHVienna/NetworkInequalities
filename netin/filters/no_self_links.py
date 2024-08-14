import numpy as np

from .filter import Filter
from ..graphs.node_vector import NodeVector

class NoSelfLinks(Filter):
    N: int

    def __init__(self, N: int) -> None:
        super().__init__()
        self.N = N

    def get_target_mask(self, source: int) -> NodeVector:
        target_mask = np.ones(self.N)
        target_mask[source] = 0.
        return NodeVector.from_ndarray(target_mask)
