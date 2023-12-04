from typing import Union, List, Dict, Tuple, Any

import numpy as np

from netin.utils import constants as const
from .pa import PA
from .g_tc import GraphTC


class PATC(GraphTC, PA):
    """Creates a new PATC instance. An undirected graph with preferential attachment and triadic closure.

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

    seed: object
        seed for random number generator

    Notes
    -----
    The initialization is an undirected graph with n nodes and no edges.
    Then, everytime a node is selected as source, it gets connected to k target nodes.
    Target nodes are selected via preferential attachment `in-degree` [BarabasiAlbert1999]_, or
    triadic closure `tc` [HolmeKim2002]_.
    """

    ############################################################
    # Constructor
    ############################################################

    def __init__(self, n: int, k: int, f_m: float, tc: float, seed: object = None):
        PA.__init__(self, n=n, k=k, f_m=f_m, seed=seed)
        GraphTC.__init__(self, n=n, k=k, f_m=f_m, tc=tc, seed=seed)
        self.model_name = const.PATC_MODEL_NAME

    ############################################################
    # Init
    ############################################################

    def validate_parameters(self):
        """
        Validates the parameters of the undirected.
        """
        PA.validate_parameters(self)
        GraphTC.validate_parameters(self)

    def get_metadata_as_dict(self) -> Dict[str, Any]:
        """
        Returns the metadata information (input parameters of the model) of the graph as a dictionary.

        Returns
        -------
        dict
            Dictionary with the graph's metadata
        """
        obj1 = PA.get_metadata_as_dict(self)
        obj2 = GraphTC.get_metadata_as_dict(self)
        obj1.update(obj2)
        return obj1

    ############################################################
    # Generation
    ############################################################

    def info_params(self):
        """
        Shows the parameters of the model.
        """
        PA.info_params(self)
        GraphTC.info_params(self)

    def get_target_probabilities_regular(self, source: int,
                                         target_list: List[int]) -> \
            Tuple[np.array, List[int]]:
        """
        Returns the probabilities of selecting a target node from a set of nodes based on the preferential attachment.

        Parameters
        ----------
        source: int
            source node

        target_list: set[int]
            set of target nodes

        special_targets: object
            special available_nodes

        Returns
        -------
        Tuple[np.array, set[int]]
            probabilities of selecting a target node from a set of nodes, and the set of target nodes

        See Also
        --------
        :py:meth:`get_target_probabilities() <PA.get_target_probabilities>` in :class:`netin.PA`.
        """
        return PA.get_target_probabilities(self, source, target_list)

    ############################################################
    # Calculations
    ############################################################

    def makecopy(self):
        """
        Makes a copy of the current object.
        """
        return self.__class__(n=self.n,
                              k=self.k,
                              f_m=self.f_m,
                              tc=self.tc,
                              seed=self.seed)

    @staticmethod
    def fit(g, n=None, k=None, seed=None):
        """
        It fits the PATC model to the given graph.

        Parameters
        ----------
        g: netin.UnDiGraph
            graph to fit the model to

        n: int
            number of nodes to override (e.g., to generate a smaller network)

        k: int
            minimum node degree to override (e.g., to generate a denser network ``k>1``)

        seed: object
            seed for random number generator

        Returns
        -------
        netin.PATC
            fitted model
        """
        n = n or g.number_of_nodes()
        k = k or g.calculate_minimum_degree()
        f_m = g.calculate_fraction_of_minority()
        tc = infer_triadic_closure(g)

        new_g = PATC(n=n,
                     k=k,
                     f_m=f_m,
                     tc=tc,
                     seed=seed)
        new_g.generate()

        return new_g

def infer_triadic_closure(g):
    import networkx as nx
    return nx.average_clustering(g)
