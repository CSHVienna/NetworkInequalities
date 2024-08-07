import numpy as np

from netin.graphs import Graph
from netin.graphs.event import Event
from .preferential_attachment import PreferentialAttachment

class InDegreePreferentialAttachment(PreferentialAttachment):
    """
    The preferential attachment link formation mechanism based on nodes' in-degree.
    """
    _a_degree: np.ndarray

    def __init__(self, graph: Graph, n: int) -> None:
        assert graph.is_directed(), "The graph must be directed."
        super().__init__(graph=graph, n=n)

        self.graph.register_event_handler(
            event=Event.LINK_ADD_AFTER,
            function=self._update_degree_by_link)

    def initialize_simulation(self):
        for i,k in self.graph.in_degree():
            self._a_degree[i] = k

    def _update_degree_by_link(self, _: int, target: int):
        """
        Updates the degree of nodes when a new link is added.

        Args:
            source (int): The source node of the new link.
            target (int): The target node of the new link.

        Returns:
            None
        """
        self._a_degree[target] += 1