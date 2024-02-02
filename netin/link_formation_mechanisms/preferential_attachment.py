import numpy as np

from netin.graphs import Graph
from netin.graphs.event import Event
from .link_formation_mechanism import LinkFormationMechanism

class PreferentialAttachment(LinkFormationMechanism):
    """
    A class representing the preferential attachment link formation mechanism.

    Parameters
    ----------
    LinkFormationMechanism : type
        The base class for link formation mechanisms.
    """
    _a_degree: np.ndarray

    def __init__(self, graph: Graph) -> None:
        """
        Initializes a PreferentialAttachment object.

        Args:
            graph (Graph): The graph object representing the network.

        Returns:
            None
        """
        super().__init__(graph)
        self._a_degree = np.zeros(len(graph), dtype=int)
        for i,k in graph.degree():
            self._a_degree[i] = k
        self.graph.register_event_handler(
            event=Event.LINK_ADD_AFTER,
            function=self._update_degree_by_link)

    def get_target_probabilities(self, _) -> np.ndarray:
        """
        Calculates the target probabilities for link formation based on the preferential attachment mechanism.

        Args:
            _ : Placeholder argument.

        Returns:
            np.ndarray: An array of target probabilities for each node in the network.
        """
        return self._a_degree / np.sum(self._a_degree)

    def _update_degree_by_link(self, source: int, target: int):
        """
        Updates the degree of nodes when a new link is added.

        Args:
            source (int): The source node of the new link.
            target (int): The target node of the new link.

        Returns:
            None
        """
        self._a_degree[source] += 1
        self._a_degree[target] += 1
