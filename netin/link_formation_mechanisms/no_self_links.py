from typing import Union
import numpy as np

from .link_formation_mechanism import LinkFormationMechanism
from ..graphs.node_attributes import NodeAttributes

class NoSelfLinks(LinkFormationMechanism):
    N: int

    def __init__(self, N: int) -> None:
        super().__init__()
        self.N = N

    def get_target_probabilities(self, source: int) -> np.ndarray:
        target_probabilities = np.ones(self.N)
        target_probabilities[source] = 0.
        return target_probabilities / target_probabilities.sum()
