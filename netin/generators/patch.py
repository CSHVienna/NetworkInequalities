from typing import Union

import numpy as np

from netin.utils import constants as const
from .pah import PAH
from .g_tc import GraphTC


class PATCH(GraphTC, PAH):
    """Creates a new PATCH instance. An undirected graph with preferential attachment, homophily, and triadic closure.

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
    Target nodes are selected via preferential attachment (in-degree) [BarabasiAlbert1999]_ and
    homophily (h_**; see :class:`netin.Homophily`) [Karimi2018]_ with probability ``1-p_{TC}``,
    and with probability ``p_{TC}`` via triadic closure (see :class:`netin.GraphTC`) [HolmeKim2002]_.

    Note that this model is still work in progress and not fully implemented yet.
    """

    ############################################################
    # Constructor
    ############################################################

    def __init__(self, n: int, k: int, f_m: float, h_mm: float, h_MM: float, tc: float, tc_uniform: bool = True, seed: object = None):
        PAH.__init__(self, n=n, k=k, f_m=f_m, h_MM=h_MM, h_mm=h_mm, seed=seed)
        GraphTC.__init__(self, n=n, k=k, f_m=f_m, tc=tc, tc_uniform=tc_uniform, seed=seed)
        self.model_name = const.PATCH_MODEL_NAME

    ############################################################
    # Init
    ############################################################

    def validate_parameters(self):
        """
        Validates the parameters of the undirected.
        """
        PAH.validate_parameters(self)
        GraphTC.validate_parameters(self)

    def get_metadata_as_dict(self) -> dict:
        """
        Returns a dictionary with the metadata of the PATCH graph.

        Returns
        -------
        dict
            the graph  metadata as a dictionary
        """
        obj1 = PAH.get_metadata_as_dict(self)
        obj2 = GraphTC.get_metadata_as_dict(self)
        obj1.update(obj2)
        return obj1

    ############################################################
    # Generation
    ############################################################
    def get_target_probabilities_regular(self, source: int,
                                         available_nodes: list[int],
                                         special_targets: Union[None, object, iter] = None) -> tuple[np.ndarray, list[int]]:
        """
        Returns the probability of nodes to be selected as target nodes using the
        preferential attachment with homophily mechanism.

        Parameters
        ----------
        source: int
            source node id

        available_nodes: set
            set of target node ids

        special_targets: dict
            dictionary of special target node ids to be considered

        Returns
        -------
        tuple
            probabilities of nodes to be selected as target nodes, and set of target of nodes
        """
        return PAH.get_target_probabilities(self, source, available_nodes, special_targets)

    ############################################################
    # Calculations
    ############################################################

    def info_params(self):
        """
        Shows the (input) parameters of the graph.
        """
        PAH.info_params(self)
        GraphTC.info_params(self)

    def info_computed(self):
        """
        Shows the (computed) properties of the graph.
        """
        PAH.info_computed(self)
        GraphTC.info_computed(self)

    def infer_homophily_values(self) -> tuple[float, float]:
        """
        Infers analytically the homophily values of the graph.

        Returns
        -------
        tuple
            homophily values of the graph (majority, minority)
        """
        h_MM = None
        h_mm = None
        return h_MM, h_mm

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
