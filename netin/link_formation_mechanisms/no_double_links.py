import numpy as np

from .link_formation_mechanism import LinkFormationMechanism
from ..graphs.graph import Graph

class NoDoubleLinks(LinkFormationMechanism):
    N: int
    graph: Graph

    def __init__(self, N: int, graph: Graph) -> None:
        super().__init__()
        self.N = N
        self.graph = graph

    def get_target_probabilities(self, source: int) -> np.ndarray:
        target_probabilities = np.ones(self.N)
        target_probabilities[self.graph[source]] = 0.
        return target_probabilities / target_probabilities.sum()
