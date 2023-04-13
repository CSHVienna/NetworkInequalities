from typing import Set
from typing import Union

import numpy as np

from netin.utils import constants as const
from .undigraph import UnDiGraph


class PA(UnDiGraph):

    ############################################################
    # Constructor
    ############################################################

    def __init__(self, n: int, k: int, f_m: float, seed: object = None):
        """

        Parameters
        ----------
        n: int
            number of nodes (minimum=2)

        k: int
            minimum degree of nodes (minimum=1)

        f_m: float
            fraction of minorities (minimum=1/n, maximum=(n-1)/n)

        Notes
        -----
        The initialization is a undigraph with n nodes and no edges.
        Then, everytime a node is selected as source, it gets connected to k target nodes.
        Target nodes are selected via preferential attachment (in-degree)

        References
        ----------
        - [1] A. L. Barabasi and R. Albert "Emergence of scaling in random networks", Science 286, pp 509-512, 1999.
        """
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
        probs = np.array([(self.degree(target) + const.EPSILON) for target in target_set])
        probs /= probs.sum()
        return probs, target_set
