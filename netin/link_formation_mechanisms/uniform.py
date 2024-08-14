import numpy as np

from .link_formation_mechanism import LinkFormationMechanism
from ..graphs.node_vector import NodeVector
from ..utils.validator import validate_int

class Uniform(LinkFormationMechanism):
    N: int

    def __init__(self, N: int) -> None:
        super().__init__()
        validate_int(N, minimum=1)
        self.N = N

    def get_target_probabilities(self, _: int) -> NodeVector:
        return NodeVector.from_ndarray(
            np.full(self.N, 1 / self.N)
        )
