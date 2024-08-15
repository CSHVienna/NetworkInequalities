from .link_formation_mechanism import LinkFormationMechanism
from ..graphs.node_vector import NodeVector

class Uniform(LinkFormationMechanism):
    def _get_target_probabilities(self, source: int) -> NodeVector:
        return self._get_uniform_target_probabilities(source)
