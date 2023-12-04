from typing import Union

import numpy as np

from netin.generators.directed import DiGraph
from netin.utils import constants as const


class DPA(DiGraph):
    """Creates a new DPA instance. A directed graph with preferential attachment.

    Parameters
    ----------
    n: int
        number of nodes (minimum=2)

    d: float
        edge density (minimum=0, maximum=1)

    f_m: float
        fraction of minorities (minimum=1/n, maximum=(n-1)/n)

    plo_M: float
        activity (out-degree power law exponent) majority group (minimum=1)

    plo_m: float
        activity (out-degree power law exponent) minority group (minimum=1)

    seed: object
        seed for random number generator

    Notes
    -----
    The initialization is a directed graph with n nodes and no edges.
    Source nodes are selected based on their activity given by plo_M (if majority) or plo_m (if minority)
    [Espin-Noboa2022]_. Target nodes are selected via preferential attachment [BarabasiAlbert1999]_.

    """

    ############################################################
    # Constructor
    ############################################################

    def __init__(self, n: int, d: float, f_m: float, plo_M: float, plo_m: float, seed: object = None):
        super().__init__(n=n, d=d, f_m=f_m, plo_M=plo_M, plo_m=plo_m, seed=seed)
        self.model_name = const.DPA_MODEL_NAME

    ############################################################
    # Generation
    ############################################################

    def get_in_degree(self, n: int) -> int:
        """
        Returns the in-degree of node `n`.
        This in-degree is not calculated, it is taken from the object `in_degrees` that is populated while
        generating the graph.

        Parameters
        ----------
        n: int
            node id

        Returns
        -------
        int
            in-degree of node `n`
        """
        return self.in_degrees[n]

    def get_target_probabilities(self, source: int, available_nodes: Union[None, list[int], np.array]) -> np.array:
        """
        Returns the probabilities for each target node in `available_nodes` to be selected as target node
        given source node `source`.

        Parameters
        ----------
        source: int
            source node id

        available_nodes: Set[int]
            set of target node ids

        Returns
        -------
        np.array
            array of probabilities for each target node.
        """
        probs = np.array([self.get_in_degree(n) + const.EPSILON for n in available_nodes])
        probs /= probs.sum()
        return probs

    def makecopy(self):
        """
        Makes a copy of the current object.
        """
        return self.__class__(n=self.n,
                              d=self.d,
                              f_m=self.f_m,
                              plo_M=self.plo_M,
                              plo_m=self.plo_m,
                              seed=self.seed)
