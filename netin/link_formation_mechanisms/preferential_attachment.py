from ..utils.constants import EPSILON
from ..graphs.graph import Graph
from ..utils.event_handling import Event
from ..graphs.node_vector import NodeVector
from .link_formation_mechanism import LinkFormationMechanism

class PreferentialAttachment(LinkFormationMechanism):
    """The preferential attachment link formation mechanism.

    Parameters
    ----------
    n : int
        The total number of nodes.
    graph : Graph
        The graph object used to update the internal degree state.
    init_degrees : bool, optional
        Whether to initialize the internal degree state, by default True
    """
    _a_degree: NodeVector

    def __init__(
            self, n: int, graph: Graph,
            init_degrees: bool = True) -> None:
        assert n >= len(graph),\
            "The number of nodes must be greater or equals to the number of nodes in the graph"
        super().__init__(n=n)
        self.graph = graph
        self._a_degree = NodeVector(
            n, dtype=int, name="degrees")

        if init_degrees:
            self.initialize_degree_array()

        self.graph.register_event_handler(
            event=Event.LINK_ADD_AFTER,
            function=self._update_degree_by_link)

    def initialize_degree_array(self):
        """Initializes the degree array.
        """
        for i in self.graph.nodes():
            self._a_degree[i] = self.graph.degree(i)

    def _get_target_probabilities(self, _: int) -> NodeVector:
        """Calculates the target probabilities for link formation
        based on the preferential attachment mechanism (relative to a node's degree).

        Parameters
        ----------
        _ : int
            Unused source node.

        Returns
        -------
        NodeVector
            An array of target probabilities for each node in the network.
        """
        a_degree_const = self._a_degree + EPSILON
        return NodeVector.from_ndarray(
            a_degree_const)

    def _update_degree_by_link(self, source: int, target: int):
        """Updates the degree of nodes when a new link is added.

        Parameters
        ----------
        source : int
            The source node of the new link.
        target : int
            The target node of the new link.
        """
        self._a_degree[source] += 1
        self._a_degree[target] += 1
