from typing import Set, Union, Tuple

import numpy as np

from netin.generators.undirected import UnDiGraph
from netin.generators.h import Homophily
from netin.generators.tc import TriadicClosure
from netin.utils import constants as const


class TCH(UnDiGraph, Homophily, TriadicClosure):
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

    def __init__(self, n: int, k: int, f_m: float, h_mm: float, h_MM: float, tc: float, seed: object = None):
        UnDiGraph.__init__(self, n, k, f_m, seed)
        Homophily.__init__(self, n=n, f_m=f_m, h_MM=h_MM, h_mm=h_mm, seed=seed)
        TriadicClosure.__init__(self, n=n, f_m=f_m, tc=tc, seed=seed)

    ############################################################
    # Init
    ############################################################

    def _infer_model_name(self):
        """
        Infers the name of the model.
        """
        return self.set_model_name(const.TCH_MODEL_NAME)

    ############################################################
    # Generation
    ############################################################

    def get_target_probabilities(self, source: Union[None, int], target_set: Union[None, Set[int]],
                                 special_targets: Union[None, object, iter] = None) -> Tuple[np.array, set[int]]:
        """
        Returns the probabilities of nodes to be selected as target nodes.

        Parameters
        ----------
        source: int
            source node id

        target_set: set
            set of target node ids

        special_targets: dict
            dictionary of special target node ids to be considered

        Returns
        -------
        tuple
            probabilities of nodes to be selected as target nodes, and set of target of nodes

        """
        return TriadicClosure.get_target_probabilities(self, source, target_set, special_targets)

    def get_target_probabilities_regular(self, source: Union[None, int], target_set: Union[None, Set[int]],
                                         special_targets: Union[None, object, iter] = None) -> Tuple[
        np.ndarray, set[int]]:
        """
        Returns the probability of nodes to be selected as target nodes using the homophily mechanism.

        Parameters
        ----------
        source: int
            source node id

        target_set: set
            set of target node ids

        special_targets: dict
            dictionary of special target node ids to be considered

        Returns
        -------
        tuple
            probabilities of nodes to be selected as target nodes, and set of target of nodes
        """
        probs = np.asarray([self.get_homophily_between_source_and_target(source, target) + const.EPSILON for target in target_set])
        return probs / probs.sum(), target_set

    def get_special_targets(self, source: int) -> object:
        """
        Returns an empty dictionary (source node ids)

        Parameters
        ----------
        source : int
            Newly added node

        Returns
        -------
        Dict
            Empty dictionary
        """
        return TriadicClosure.get_special_targets(self, source)

    ############################################################
    # Calculations
    ############################################################

    def info_params(self):
        """
        Shows the (input) parameters of the graph.
        """
        Homophily.info_params(self)
        TriadicClosure.info_params(self)

    def info_computed(self):
        """
        Shows the (computed) properties of the graph.
        """
        Homophily.info_computed(self)
        TriadicClosure.info_computed(self)

    def infer_homophily_values(self) -> Tuple[float, float]:
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
        tc = None
        return tc

    def _makecopy(self):
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
