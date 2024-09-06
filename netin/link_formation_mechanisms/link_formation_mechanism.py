from abc import abstractmethod

import numpy as np

from ..utils.validator import validate_int
from ..base_class import BaseClass
from ..graphs.node_vector import NodeVector

class LinkFormationMechanism(BaseClass):
    """Base class for link formation mechanisms.

    Parameters
    ----------
    N : int
        Number of nodes.
    """
    N: int

    def __init__(self, N: int) -> None:
        super().__init__()
        validate_int(N, minimum=1)
        self.N = N

    def get_target_probabilities(self, source: int) -> NodeVector:
        """Returns the probabilities to connect to target nodes.
        The values are checked to be non-negative and sum to 1.
        Note that a uniform distribution is returned if all probabilities are zero.

        Parameters
        ----------
        source : int
            The source node.

        Returns
        -------
        NodeVector
            The probabilities to connect to the target nodes.
        """
        probs = self._get_target_probabilities(source)
        if not np.any(probs != 0.):
            probs = self._get_uniform_target_probabilities(source)
        assert not np.any(np.isnan(probs)), "Probabilities contain NaN values."
        assert not np.any(probs < 0), "Probabilities must be non-negative."
        return probs / np.sum(probs)

    @abstractmethod
    def _get_target_probabilities(self, source: int) -> NodeVector:
        """Returns the probabilities to connect to target nodes.
        Should be implemented by the actual link formation mechanism subclasses.

        Parameters
        ----------
        source : int
            The source node.

        Returns
        -------
        NodeVector
            The probabilities to connect to the target nodes.
        """
        raise NotImplementedError

    def _get_uniform_target_probabilities(self, _: int) -> NodeVector:
        return NodeVector.from_ndarray(
            np.full(self.N, 1 / self.N)
        )
