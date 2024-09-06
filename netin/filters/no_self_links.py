import numpy as np

from .filter import Filter
from ..graphs.node_vector import NodeVector

class NoSelfLinks(Filter):
    """Prevents self-links in the graph.

    Parameters
    ----------
    N : int
        Number of nodes.
    """
    N: int

    def __init__(self, N: int) -> None:
        super().__init__()
        self.N = N

    def get_target_mask(self, source: int) -> NodeVector:
        """Returns 0. for the source node and 1. for all other nodes.

        Parameters
        ----------
        source : int
            The source node.

        Returns
        -------
        NodeVector
            The filter mask for the target nodes.
        """
        target_mask = np.ones(self.N)
        target_mask[source] = 0.
        return NodeVector.from_ndarray(target_mask)
