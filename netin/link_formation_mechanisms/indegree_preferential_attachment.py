from .preferential_attachment import PreferentialAttachment
from ..graphs.directed import DiGraph
from ..graphs.node_vector import NodeVector

class InDegreePreferentialAttachment(PreferentialAttachment):
    """The preferential attachment link formation mechanism based on nodes' in-degree.

    Parameters
    ----------
    graph : DiGraph
        The graph which is used to update the in-degrees.
    N : int
        Number of nodes.
    init_degrees : bool, optional
        Whether to initialize the in-degrees from the provided `graph`, by default True
    """
    _a_degree: NodeVector

    def __init__(self,
                 graph: DiGraph,
                 N: int,
                 init_degrees: bool = True) -> None:
        assert graph.is_directed(), "The graph must be directed."
        super().__init__(N=N, graph=graph)

        self._a_degree = NodeVector(
            N, dtype=int, name="in_degrees")
        if init_degrees:
            self.initialize_degree_array()

    def initialize_degree_array(self):
        """Initializes the degree array.
        This is useful for when an existing graph is loaded and the degrees need to be computed.
        """
        # @TODO: This is slow.
        # One could optimize this by storing incoming links in the graph object.
        for i in self.graph.nodes():
            _in_degree = 0
            for j in self.graph.nodes():
                if i != j and self.graph.has_edge(j, i):
                    _in_degree += 1
            self._a_degree[i] = _in_degree

    def _update_degree_by_link(self, _: int, target: int):
        """Updates the degree of nodes when a new link is added.

        Parameters
        ----------
        _ : int
            Unused.
        target : int
            The target node.
        """
        self._a_degree[target] += 1
