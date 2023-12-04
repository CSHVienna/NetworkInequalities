from typing import Union

import numpy as np

from netin.generators.undirected import UnDiGraph
from netin.utils import constants as const


class PA(UnDiGraph):
    """Creates a new PA instance. An undirected graph with preferential attachment.

    Parameters
    ----------
    n: int
        number of nodes (minimum=2)

    k: int
        minimum degree of nodes (minimum=1)

    f_m: float
        fraction of minorities (minimum=1/n, maximum=(n-1)/n)

    seed: object
        seed for random number generator

    Notes
    -----
    The initialization is an undirected graph with n nodes and no edges.
    Then, everytime a node is selected as source, it gets connected to k target nodes.
    Target nodes are selected via preferential attachment (in-degree), see [BarabasiAlbert1999]_.
    """

    ############################################################
    # Constructor
    ############################################################

    def __init__(self, n: int, k: int, f_m: float, seed: object = None):
        super().__init__(n, k, f_m, seed)
        self.model_name = const.PA_MODEL_NAME

    ############################################################
    # Generation
    ############################################################

    def get_target_probabilities(self, source: int, available_nodes: list[int]) -> tuple[np.array, list[int]]:
        """
        Returns the probabilities of the target nodes to be selected given a source node.
        This probability is proportional to the degree of the target node.

        Parameters
        ----------
        source: int
            source node (id)

        available_nodes: set
            set of target nodes (ids)

        special_targets: object
            special available_nodes

        Returns
        -------
        probs: np.array
            probabilities of the target nodes to be selected

        available_nodes: set
            set of target nodes (ids)
        """
        probs = np.array([(self.degree(target) + const.EPSILON) for target in available_nodes])
        probs /= probs.sum()
        return probs, available_nodes

    def makecopy(self):
        """
        Makes a copy of the current object.
        """
        return self.__class__(n=self.n,
                              k=self.k,
                              f_m=self.f_m,
                              seed=self.seed)

    @staticmethod
    def fit(g, n=None, k=None, seed=None):
        """
        It fits the PA model to the given graph.

        Parameters
        ----------
        g: netin.UnDiGraph
            graph to fit the model to

        n: int
            number of nodes to override (e.g., to generate a smaller network)

        k: int
            minimum node degree to override (e.g., to generate a denser network ``k>1``)

        seed: object
            seed for random number generator

        Returns
        -------
        netin.PA
            fitted model
        """
        n = n or g.number_of_nodes()
        k = k or g.calculate_minimum_degree()
        f_m = g.calculate_fraction_of_minority()

        new_g = PA(n=n,
                   k=k,
                   f_m=f_m,
                   seed=seed)
        new_g.generate()

        return new_g
