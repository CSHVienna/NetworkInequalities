import numpy as np

from .filter import Filter
from ..graphs.node_vector import NodeVector

class NoSelfLinks(Filter):
    """Prevents self-links in the graph.

    Parameters
    ----------
    n : int
        Number of nodes.
    """
    n: int

    def __init__(self, n: int) -> None:
        super().__init__()
        self.n = n

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
        target_mask = np.ones(self.n)
        target_mask[source] = 0.
        return NodeVector.from_ndarray(target_mask)
