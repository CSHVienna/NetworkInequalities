from typing import Union, Set

import numpy as np

from netin.utils import constants as const
from .digraph import DiGraph
from .h import Homophily


class DH(DiGraph, Homophily):

    ############################################################
    # Constructor
    ############################################################

    def __init__(self, n: int, d: float, f_m: float, plo_M: float, plo_m: float, h_MM: float, h_mm: float,
                 seed: object = None):
        """

        Parameters
        ----------
        n: int
            number of nodes (minimum=2)

        k: int
            minimum degree of nodes (minimum=1)

        f_m: float
            fraction of minorities (minimum=1/n, maximum=(n-1)/n)

        d: float
            edge density (minimum=0, maximum=1)

        plo_M: float
            activity (out-degree power law exponent) majority group (minimum=1)

        plo_m: float
            activity (out-degree power law exponent) minority group (minimum=1)

        h_MM: float
            homophily within majority group (minimum=0, maximum=1)

        h_mm: float
            homophily within minority group (minimum=0, maximum=1)

        seed: object
            seed for random number generator

        Notes
        -----
        The initialization is a digraph with n nodes and no edges.
        Then, everytime a node is selected as source, it gets connected to k target nodes.
        Target nodes are selected via preferential attachment (in-degree)

        References
        ----------
        - [1] A. L. Barabasi and R. Albert "Emergence of scaling in random networks", Science 286, pp 509-512, 1999.
        """
        DiGraph.__init__(self, n=n, d=d, f_m=f_m, plo_M=plo_M, plo_m=plo_m, seed=seed)
        Homophily.__init__(self, n=n, f_m=f_m, h_MM=h_MM, h_mm=h_mm, seed=seed)

    ############################################################
    # Init
    ############################################################

    def _infer_model_name(self):
        """
        Infers the name of the model.
        """
        return self.set_model_name(const.DH_MODEL_NAME)

    ############################################################
    # Generation
    ############################################################

    def _initialize(self, class_attribute: str = 'm', class_values: list = None, class_labels: list = None):
        DiGraph._initialize(self, class_attribute, class_values, class_labels)
        Homophily._initialize(self, class_attribute, class_values, class_labels)

    def get_target_probabilities(self, source: Union[None, int], target_set: Union[None, Set[int], np.array],
                                 special_targets: Union[None, object, iter] = None) -> np.array:
        probs, ts = Homophily.get_target_probabilities(self, source, target_set, special_targets)
        return probs

    ############################################################
    # Calculations
    ############################################################

    def info_params(self):
        DiGraph.info_params(self)
        Homophily.info_params(self)