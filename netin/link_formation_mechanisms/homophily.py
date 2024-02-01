import numpy as np

from netin.graphs import Graph
from .link_formation_mechanism import LinkFormationMechanism

class Homophily(LinkFormationMechanism):
    _a_class: np.ndarray
    _a_h: np.ndarray
    h: float

    def __init__(self, graph: Graph, h: float) -> None:
        super().__init__(graph)
        self.h=h
        self._a_class = np.full(len(self.graph), -1)
        for node in self.graph:
            self._a_class[node] = self.graph.node_class_values[node]

    def get_target_probabilities(self, source: int) -> np.ndarray:
        class_source = self.graph.node_class_values[source]
        return np.where(class_source == self._a_class, self.h, 1 - self.h)
