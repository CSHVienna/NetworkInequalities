from typing import Any, List, Union

import numpy as np

from netin.generators.h import Homophily
from netin.utils import constants as const

from .g_tc import GraphTC

class TCH(GraphTC, Homophily):
    """Creates a new TCH graph. An undirected graph with homophily and triadic closure as link formation mechanisms.

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

    tc_uniform: bool
        specifies whether the triadic closure target is chosen uniform at random or if it follows the regular link formation mechanisms (e.g., homophily) (default=True)

    Notes
    -----
    The initialization is an undirected graph with n nodes and no edges.
    Then, everytime a node is selected as source, it gets connected to k target nodes.
    Target nodes are selected via homophily (h_**; see :class:`netin.Homophily`) [Karimi2018]_ with probability ``1-p_{TC}``,
    and with probability ``p_{TC}`` via triadic closure (see :class:`netin.TriadicClosure`) [HolmeKim2002]_.

    Note that this model is still work in progress and not fully implemented yet.
    """

    ############################################################
    # Constructor
    ############################################################

    def __init__(self, n: int, k: int, f_m: float, h_mm: float, h_MM: float, tc: float, tc_uniform: bool = True,
                 seed: object = None):
        GraphTC.__init__(self, n=n, k=k, f_m=f_m, tc=tc, tc_uniform=tc_uniform, seed=seed)
        Homophily.__init__(self, n=n, f_m=f_m, h_MM=h_MM, h_mm=h_mm, seed=seed)
        self.model_name = const.TCH_MODEL_NAME

    ############################################################
    # Generation
    ############################################################
    def get_target_probabilities_regular(self, source: int, target_list: list[int]) -> \
            tuple[np.ndarray, list[int]]:
        """
        Returns the probability of nodes to be selected as target nodes using the homophily mechanism.

        Parameters
        ----------
        source: int
            source node id

        target_list: set
            set of target node ids

        special_targets: dict
            dictionary of special target node ids to be considered

        Returns
        -------
        tuple
            probabilities of nodes to be selected as target nodes, and set of target of nodes
        """
        return Homophily.get_target_probabilities(self=self, source=source, available_nodes=target_list)

    def initialize(self, class_attribute: str = 'm', class_values: List[Any] = None, class_labels: List[str] = None):
        return Homophily.initialize(class_attribute, class_values, class_labels)

    ############################################################
    # Calculations
    ############################################################

    def info_params(self):
        """
        Shows the (input) parameters of the graph.
        """
        Homophily.info_params(self)
        GraphTC.info_params(self)

    def info_computed(self):
        """
        Shows the (computed) properties of the graph.
        """
        Homophily.info_computed(self)
        GraphTC.info_computed(self)

    def infer_triadic_closure(self) -> float:
        """
        Infers analytically the triadic closure value of the graph.
        @TODO: This still needs to be implemented.
        Returns
        -------
        float
            triadic closure probability of the graph
        """
        raise NotImplementedError("Inferring triadic closure probability not implemented yet.")

    def makecopy(self):
        """
        Makes a copy of the current object.
        """
        return self.__class__(n=self.n,
                              k=self.k,
                              f_m=self.f_m,
                              tc=self.tc,
                              h_MM=self.h_MM,
                              h_mm=self.h_mm,
                              seed=self.seed)
