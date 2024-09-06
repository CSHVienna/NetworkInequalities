from abc import abstractmethod

from ..base_class import BaseClass
from ..graphs.node_vector import NodeVector

class Filter(BaseClass):
    """Base class for filters."""

    @abstractmethod
    def get_target_mask(self, source: int) -> NodeVector:
        """Returns a mask for the target nodes.

        Parameters
        ----------
        source : int
            The source node.

        Returns
        -------
        NodeVector
            A filter mask for the target nodes. Values must be 0. or 1.
        """
        raise NotImplementedError
