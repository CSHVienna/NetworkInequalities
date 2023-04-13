from typing import Union, Set

import numpy as np

from netin.utils import constants as const
from .h import Homophily
from .pa import PA


class PAH(PA, Homophily):

    ############################################################
    # Constructor
    ############################################################

    def __init__(self, n: int, k: int, f_m: float, h_MM: float, h_mm: float, seed: object = None):
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

        Notes
        -----
        The initialization is a undigraph with n nodes and no edges.
        Then, everytime a node is selected as source, it gets connected to k target nodes.
        Target nodes are selected via preferential attachment (in-degree), and homophily (h_**)

        References
        ----------
        - [1] A. L. Barabasi and R. Albert "Emergence of scaling in random networks", Science 286, pp 509-512, 1999.
        """
        PA.__init__(self, n=n, k=k, f_m=f_m, seed=seed)
        Homophily.__init__(self, n=n, f_m=f_m, h_MM=h_MM, h_mm=h_mm, seed=seed)

    ############################################################
    # Init
    ############################################################

    def _infer_model_name(self):
        """
        Infers the name of the model.
        """
        return self.set_model_name(const.PAH_MODEL_NAME)

    def _validate_parameters(self):
        """
        Validates the parameters of the undigraph.
        """
        PA._validate_parameters(self)
        Homophily._validate_parameters(self)

    def get_metadata_as_dict(self) -> dict:
        obj = PA.get_metadata_as_dict(self)
        obj.update(Homophily.get_metadata_as_dict(self))
        return obj

    ############################################################
    # Generation
    ############################################################

    def _initialize(self, class_attribute: str = 'm', class_values: list = None, class_labels: list = None):
        PA._initialize(self, class_attribute, class_values, class_labels)
        Homophily._initialize(self, class_attribute, class_values, class_labels)

    def get_target_probabilities(self, source: Union[None, int], target_set: Union[None, Set[int]],
                                 special_targets: Union[None, object, iter] = None) -> tuple[np.array, set[int]]:
        probs = np.array([self.get_homophily_between_source_and_target(source, target) *
                          (self.degree(target) + const.EPSILON) for target in target_set])
        probs /= probs.sum()
        return probs, target_set

    ############################################################
    # Calculations
    ############################################################

    def info_params(self):
        PA.info_params(self)
        Homophily.info_params(self)

    def infer_homophily_values(self) -> (float, float):
        # @TODO: To be implemented
        return None, None
