import numpy as np

from netin.graphs import Graph
from netin.graphs.event import Event
from .link_formation_mechanism import LinkFormationMechanism

class PreferentialAttachment(LinkFormationMechanism):
    _a_degree: np.ndarray

    def __init__(self, graph: Graph) -> None:
        super().__init__(graph)
        self._a_degree = np.zeros(len(graph), dtype=int)
        for i,k in graph.degree():
            self._a_degree[i] = k
        self.graph.register_event_handler(
            event=Event.LINK_ADD_AFTER,
            function=self._update_degree_by_link)

    def get_target_probabilities(self, _) -> np.ndarray:
        return self._a_degree / np.sum(self._a_degree)

    def _update_degree_by_link(self, source: int, target: int):
        self._a_degree[source] += 1
        self._a_degree[target] += 1
