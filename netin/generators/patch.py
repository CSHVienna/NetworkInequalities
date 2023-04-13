from typing import Union, Set

import numpy as np

from netin.utils import constants as const
from .pah import PAH
from .tc import TC


class PATCH(PAH, TC):

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
        PAH.__init__(self, n=n, k=k, f_m=f_m, h_MM=h_MM, h_mm=h_mm, seed=seed)
        TC.__init__(self, n=n, f_m=f_m, tc=tc, seed=seed)

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
        PAH._validate_parameters(self)
        TC._validate_parameters(self)

    def get_metadata_as_dict(self) -> dict:
        obj1 = PAH.get_metadata_as_dict(self)
        obj2 = TC.get_metadata_as_dict(self)
        obj1.update(obj2)
        return obj1

    ############################################################
    # Generation
    ############################################################

    def get_target_probabilities(self, source: Union[None, int], target_set: Union[None, Set[int]],
                                 special_targets: Union[None, object, iter] = None) -> tuple[np.array, set[int]]:
        return TC.get_target_probabilities(self, source, target_set, special_targets)

    def get_target_probabilities_regular(self, source: Union[None, int], target_set: Union[None, Set[int]],
                                         special_targets: Union[None, object, iter] = None) -> tuple[
        np.ndarray, set[int]]:
        return PAH.get_target_probabilities(self, source, target_set, special_targets)

    def get_special_targets(self, source: int) -> object:
        """
        Return an empty dictionary (source node ids)
        Parameters
        ----------
        source: int
            Newly added node
        """
        return TC.get_special_targets(self, source)

    ############################################################
    # Calculations
    ############################################################

    def info_params(self):
        PAH.info_params(self)
        TC.info_params(self)

    def info_computed(self):
        PAH.info_computed(self)
        TC.info_computed(self)
