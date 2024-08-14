from abc import abstractmethod

from ..base_class import BaseClass
from ..graphs.node_vector import NodeVector

class LinkFormationMechanism(BaseClass):
    def get_target_probabilities(self, source: int) -> NodeVector:
        probs = self._get_target_probabilities(source)
        assert probs.sum == 1., "Probabilities do not sum to 1."
        return probs

    @abstractmethod
    def _get_target_probabilities(self, source: int) -> NodeVector:
        raise NotImplementedError
