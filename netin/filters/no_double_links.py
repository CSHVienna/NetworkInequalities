import numpy as np

from .filter import Filter
from ..graphs.graph import Graph
from ..graphs.node_vector import NodeVector

class NoDoubleLinks(Filter):
    """Prevents double links in the graph.

    Parameters
    ----------
    n : int
        Number of nodes.
    graph : Graph
        The graph used to update node activity.
    """
    n: int
    graph: Graph

    def __init__(self, n: int, graph: Graph) -> None:
        super().__init__()
        self.n = n
        self.graph = graph

    def get_target_mask(self, source: int) -> NodeVector:
        """Returns 0. for nodes that are already linked to the source node and 1. otherwise.

        Parameters
        ----------
        source : int
            The source node.

        Returns
        -------
        NodeVector
            The filter mask for the target nodes.
        """
        target_mask = np.ones(self.n)
        target_mask[list(self.graph[source])] = 0.
        return NodeVector.from_ndarray(target_mask)
