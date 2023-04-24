from typing import Set
from typing import Union

import numpy as np

from netin.utils import constants as const
from netin.generators.undirected import UnDiGraph


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

    ############################################################
    # Init
    ############################################################

    def _infer_model_name(self):
        """
        Infers the name of the model.
        """
        return self.set_model_name(const.PA_MODEL_NAME)

    ############################################################
    # Generation
    ############################################################

    def get_target_probabilities(self, source: Union[None, int], target_set: Union[None, Set[int]],
                                 special_targets: Union[None, object, iter] = None) -> tuple[np.array, set[int]]:
        """
        Returns the probabilities of the target nodes to be selected given a source node.
        This probability is proportional to the degree of the target node.

        Parameters
        ----------
        source: int
            source node (id)

        target_set: set
            set of target nodes (ids)

        special_targets: object
            special targets

        Returns
        -------
        probs: np.array
            probabilities of the target nodes to be selected

        target_set: set
            set of target nodes (ids)
        """
        probs = np.array([(self.degree(target) + const.EPSILON) for target in target_set])
        probs /= probs.sum()
        return probs, target_set

    def _makecopy(self):
        """
        Makes a copy of the current object.
        """
        return self.__class__(n=self.n,
                              k=self.k,
                              f_m=self.f_m,
                              seed=self.seed)
