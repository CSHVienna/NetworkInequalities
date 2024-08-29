from abc import abstractmethod

import numpy as np

from ..utils.validator import validate_int
from ..utils.constants import EPSILON
from ..base_class import BaseClass
from ..graphs.node_vector import NodeVector

class LinkFormationMechanism(BaseClass):
    N: int

    def __init__(self, N: int) -> None:
        super().__init__()
        validate_int(N, minimum=1)
        self.N = N

    def get_target_probabilities(self, source: int) -> NodeVector:
        probs = self._get_target_probabilities(source)
        if not np.any(probs != 0.):
            probs = self._get_uniform_target_probabilities(source)
        assert not np.any(np.isnan(probs)), "Probabilities contain NaN values."
        return probs / np.sum(probs)

    @abstractmethod
    def _get_target_probabilities(self, source: int) -> NodeVector:
        raise NotImplementedError

    def _get_uniform_target_probabilities(self, _: int) -> NodeVector:
        return NodeVector.from_ndarray(
            np.full(self.N, 1 / self.N)
        )
