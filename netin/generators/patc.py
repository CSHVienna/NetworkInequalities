from typing import Union, Set

import numpy as np

from netin.utils import constants as const
from netin.generators.tc import TriadicClosure
from .pa import PA


class PATC(PA, TriadicClosure):

    ############################################################
    # Constructor
    ############################################################

    def __init__(self, n: int, k: int, f_m: float, tc: float, seed: object = None):
        """

        Parameters
        ----------
        n: int
            number of nodes (minimum=2)

        k: int
            minimum degree of nodes (minimum=1)

        f_m: float
            fraction of minorities (minimum=1/n, maximum=(n-1)/n)

        tc: float
            probability of a new edge to close a triad (minimum=0, maximum=1.)

        attr: dict
            attributes to add to undirected as key=value pairs

        Notes
        -----
        The initialization is a undirected with n nodes and no edges.
        Then, everytime a node is selected as source, it gets connected to k target nodes.
        Target nodes are selected via preferential attachment (in-degree), and homophily (h_**)

        References
        ----------
        - [1] A. L. Barabasi and R. Albert "Emergence of scaling in random networks", Science 286, pp 509-512, 1999.
        """
        PA.__init__(self, n=n, k=k, f_m=f_m, seed=seed)
        TriadicClosure.__init__(self, n=n, f_m=f_m, tc=tc, seed=seed)

    ############################################################
    # Init
    ############################################################

    def _infer_model_name(self):
        """
        Infers the name of the model.
        """
        return self.set_model_name(const.PATC_MODEL_NAME)

    def _validate_parameters(self):
        """
        Validates the parameters of the undirected.
        """
        PA._validate_parameters(self)
        TriadicClosure._validate_parameters(self)

    def get_metadata_as_dict(self) -> dict:
        obj1 = PA.get_metadata_as_dict(self)
        obj2 = TriadicClosure.get_metadata_as_dict(self)
        obj1.update(obj2)
        return obj1

    ############################################################
    # Generation
    ############################################################

    def info_params(self):
        PA.info_params(self)
        TriadicClosure.info_params(self)

    def get_special_targets(self, source: int) -> object:
        """
        Return an empty dictionary (source node ids)
        Parameters
        ----------
        source: int
            Newly added node
        """
        return TriadicClosure.get_special_targets(self, source)

    def get_target_probabilities(self, source: Union[None, int], target_set: Union[None, Set[int]],
                                 special_targets: Union[None, object, iter] = None) -> tuple[np.array, set[int]]:
        return TriadicClosure.get_target_probabilities(self, source, target_set, special_targets)

    def get_target_probabilities_regular(self, source: Union[None, int], target_set: Union[None, Set[int]],
                                         special_targets: Union[None, object, iter] = None) -> tuple[
        np.array, set[int]]:
        return PA.get_target_probabilities(self, source, target_set, special_targets)

    def update_special_targets(self, idx_target: int, source: int, target: int, targets: Set[int],
                               special_targets: object) -> object:
        return TriadicClosure.update_special_targets(self, idx_target, source, target, targets, special_targets)
