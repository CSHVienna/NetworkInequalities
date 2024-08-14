import numpy as np

from ..utils.constants import EPSILON
from ..graphs.graph import Graph
from ..graphs.event import Event
from ..graphs.node_vector import NodeVector
from ..utils.validator import validate_int
from .link_formation_mechanism import LinkFormationMechanism

class PreferentialAttachment(LinkFormationMechanism):
    """
    A class representing the preferential attachment link formation mechanism.

    Parameters
    ----------
    LinkFormationMechanism : type
        The base class for link formation mechanisms.
    """
    _a_degree: NodeVector

    def __init__(
            self, graph: Graph, n: int,
            init_degrees: bool = True) -> None:
        """
        Initializes a PreferentialAttachment object.

        Args:
            graph (Graph): The graph object representing the network.
            n (int): The total (final) number of nodes.
            init_degrees (bool, optional): Whether to initialize the degree array. Defaults to True.

        Returns:
            None
        """
        validate_int(n, minimum=1)
        super().__init__()
        self.graph = graph
        self._a_degree = NodeVector(
            n, dtype=int, name="degrees")

        if init_degrees:
            self.initialize_degree_array()

        self.graph.register_event_handler(
            event=Event.LINK_ADD_AFTER,
            function=self._update_degree_by_link)

    def initialize_degree_array(self):
        for i,k in self.graph.degree():
            self._a_degree[i] = k

    def get_target_probabilities(self, _) -> NodeVector:
        """
        Calculates the target probabilities for link formation based on the preferential attachment mechanism.

        Args:
            _ : Placeholder argument.

        Returns:
            NodeVector: An array of target probabilities for each node in the network.
        """
        a_degree_const = self._a_degree + EPSILON
        return NodeVector.from_ndarray(
            a_degree_const / np.sum(a_degree_const))

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
