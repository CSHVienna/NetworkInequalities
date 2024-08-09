from typing import Union, Optional
import numpy as np

from .link_formation_mechanism import LinkFormationMechanism
from ..graphs.node_attributes import NodeAttributes

class Uniform(LinkFormationMechanism):
    N: int

    def __init__(self, N: int) -> None:
        super().__init__()
        self.N = N

    def get_target_probabilities(self, _: int) -> NodeAttributes:
        return NodeAttributes.from_ndarray(
            np.full(self.N, 1 / self.N)
        )
