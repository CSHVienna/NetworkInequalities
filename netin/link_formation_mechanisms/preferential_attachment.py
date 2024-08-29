import numpy as np

from ..utils.constants import EPSILON
from ..graphs.graph import Graph
from ..utils.event_handling import Event
from ..graphs.node_vector import NodeVector
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
            self, N: int, graph: Graph,
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
        assert N >= len(graph),\
            "The number of nodes must be greater or equals to the number of nodes in the graph"
        super().__init__(N=N)
        self.graph = graph
        self._a_degree = NodeVector(
            N, dtype=int, name="degrees")

        if init_degrees:
            self.initialize_degree_array()

    def initialize_degree_array(self):
        for i in self.graph.nodes():
            self._a_degree[i] = self.graph.degree(i)

    def _get_target_probabilities(self, _) -> NodeVector:
        """
        Calculates the target probabilities for link formation based on the preferential attachment mechanism.

        Args:
            _ : Placeholder argument.

        Returns:
            NodeVector: An array of target probabilities for each node in the network.
        """
        a_degree_const = self._a_degree + EPSILON
        return NodeVector.from_ndarray(
            a_degree_const)

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
