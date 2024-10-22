from typing import List, Tuple

import numpy as np

from netin.utils import constants as const
from .patch import PATCH
from .pah import PAH
from .graph import Graph


class ERPATCH(PATCH):
    """Creates a new ERPATCH instance. This graph choses the initial link randomly (following Erdos-Renyi, ER) and biases the triadic closure (TC) candidate selection based on preferential attachment (PA) and homophily (H).

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
    The initialization is an undirected graph with n nodes and no edges.
    Then, everytime a node is selected as source, it gets connected to k target nodes.
    Target nodes are selected via preferential attachment (in-degree) [BarabasiAlbert1999]_ and
    homophily (h_**; see :class:`netin.Homophily`) [Karimi2018]_ with probability ``1-p_{TC}``,
    and with probability ``p_{TC}`` via triadic closure (see :class:`netin.GraphTC`) [HolmeKim2002]_.

    Note that this model is still work in progress and not fully implemented yet.
    """
    def __init__(self, n: int, k: int, f_m: float, h_mm: float, h_MM: float, tc: float, tc_uniform: bool = False, seed: object = None):
        super().__init__(n=n, k=k, f_m=f_m, h_mm=h_mm, h_MM=h_MM, tc=tc, tc_uniform=tc_uniform, seed=seed)
        self.model_name = const.ERPATCH_MODEL_NAME

    ############################################################
    # Generation
    ############################################################
    def get_target_probabilities_regular(self, source: int,
                                         target_list: List[int]) -> Tuple[np.ndarray, List[int]]:
        return Graph.get_target_probabilities(self, source, target_list)

    def get_target_probabilities_tc(self, source: int, target_list: List[int]) -> Tuple[np.ndarray, List[int]]:
        return PAH.get_target_probabilities(self, source, target_list)

    def infer_triadic_closure(self) -> float:
        """
        Infers analytically the triadic closure value of the graph.

        Returns
        -------
        float
            triadic closure probability of the graph
        """
        # @TODO: To be implemented
        raise NotImplementedError("Inferring triadic closure not implemented yet.")
