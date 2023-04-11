from typing import Union, Set

import numpy as np

from netin.utils import constants as const
from .pah import PAH
from .patc import PATC


class PATCH(PATC, PAH):

    ############################################################
    # Constructor
    ############################################################

    def __init__(self, n: int, k: int, f_m: float, h_mm: float, h_MM: float, tc: float, seed: object = None):
        """

        Parameters
        ----------
        n: int
            number of nodes (minimum=2)

        k: int
            minimum degree of nodes (minimum=1)

        f_m: float
            fraction of minorities (minimum=1/n, maximum=(n-1)/n)

        h_MM: float
            homophily (similarity) between majority nodes (minimum=0, maximum=1.)

        h_mm: float
            homophily (similarity) between minority nodes (minimum=0, maximum=1.)

        tc: float
            probability of a new edge to close a triad (minimum=0, maximum=1.)

        Notes
        -----
        The initialization is a undigraph with n nodes and no edges.
        Then, everytime a node is selected as source, it gets connected to k target nodes.
        Target nodes are selected via preferential attachment (in-degree), and homophily (h_**)

        References
        ----------
        - [1] A. L. Barabasi and R. Albert "Emergence of scaling in random networks", Science 286, pp 509-512, 1999.
        """
        PATC.__init__(self, n, k, f_m, tc, seed)
        PAH.__init__(self, n, k, f_m, h_MM, h_mm, seed)

    ############################################################
    # Init
    ############################################################

    def _infer_model_name(self):
        """
        Infers the name of the model.
        """
        return self.set_model_name(const.PATCH_MODEL_NAME)

    def _validate_parameters(self):
        """
        Validates the parameters of the undigraph.
        """
        PATC._validate_parameters(self)
        PAH._validate_parameters(self)

    def get_metadata_as_dict(self) -> dict:
        obj1 = PATC.get_metadata_as_dict(self)
        obj2 = PAH.get_metadata_as_dict(self)
        obj1.update(obj2)
        return obj1

    ############################################################
    # Generation
    ############################################################

    def get_target_probabilities_regular(self, source: Union[None, int], target_set: Union[None, Set[int]],
                                         special_targets: Union[None, object, iter] = None) -> tuple[
        np.ndarray, set[int]]:
        return PAH.get_target_probabilities(self, source, target_set, special_targets)

    ############################################################
    # Calculations
    ############################################################

    def info_params(self):
        PATC.info_params(self)
        PAH.info_params(self)

    def info_computed(self):
        PATC.info_computed(self)
        PAH.info_computed(self)
