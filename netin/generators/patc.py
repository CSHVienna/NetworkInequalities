from typing import Union, List, Dict

import numpy as np

from netin.generators.tc import TriadicClosure
from netin.utils import constants as const
from .pa import PA


class PATC(PA, TriadicClosure):
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
        TriadicClosure.__init__(self, n=n, f_m=f_m, tc=tc, seed=seed)
        self.model_name = const.PATC_MODEL_NAME

    ############################################################
    # Init
    ############################################################

    def validate_parameters(self):
        """
        Validates the parameters of the undirected.
        """
        PA.validate_parameters(self)
        TriadicClosure.validate_parameters(self)

    def get_metadata_as_dict(self) -> dict:
        """
        Returns the metadata information (input parameters of the model) of the graph as a dictionary.

        Returns
        -------
        dict
            Dictionary with the graph's metadata
        """
        obj1 = PA.get_metadata_as_dict(self)
        obj2 = TriadicClosure.get_metadata_as_dict(self)
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
        TriadicClosure.info_params(self)

    def get_special_targets(self, source: int) -> object:
        """
        Returns the initial set of special available_nodes based on the triadic closure mechanism.

        Parameters
        ----------
        source : int
             Newly added node

        Returns
        -------
        object
            Empty dictionary (source node ids)

        See Also
        --------
        :py:meth:`get_special_targets() <TriadicClosure.get_special_targets>` in :class:`netin.TC`.
        """
        return TriadicClosure.get_special_targets(self, source)

    def get_target_probabilities(self, source: int, available_nodes: list[int],
                                 special_targets: Union[None, object, iter] = None) -> tuple[np.array, list[int]]:
        """
        Returns the probabilities of selecting a target node from a set of nodes based on the preferential attachment
        or triadic closure.

        Parameters
        ----------
        source: int
            source node

        available_nodes: set[int]
            set of target nodes

        special_targets: object
            special available_nodes

        Returns
        -------
        tuple[np.array, set[int]]
            probabilities of selecting a target node from a set of nodes, and the set of target nodes

        See Also
        --------
        :py:meth:`get_target_probabilities() <TriadicClosure.get_target_probabilities>` in :class:`netin.TC`.
        """
        return TriadicClosure.get_target_probabilities(self, source, available_nodes, special_targets)

    def get_target_probabilities_regular(self, source: int,
                                         target_list: list[int],
                                         special_targets: Union[None, object, iter] = None) -> \
            tuple[np.array, list[int]]:
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
        tuple[np.array, set[int]]
            probabilities of selecting a target node from a set of nodes, and the set of target nodes

        See Also
        --------
        :py:meth:`get_target_probabilities() <PA.get_target_probabilities>` in :class:`netin.PA`.
        """
        return PA.get_target_probabilities(self, source, target_list, special_targets)

    def update_special_targets(self, idx_target: int,
                               source: int, target: int,
                               available_nodes: List[int],
                               special_targets: Union[None, Dict[int, int]]) ->  Union[None, Dict[int, int]]:
        """
        Updates the set of special available_nodes based on the triadic closure mechanism.

        Parameters
        ----------
        idx_target: int
            index of the target node

        source: int
            source node

        target: int
            target node

        available_nodes: Set[int]
            set of target nodes

        special_targets: object
            special available_nodes

        Returns
        -------
        object
            updated special available_nodes

        See Also
        --------
        :func:`netin.TC.update_special_targets`

        """
        return TriadicClosure.update_special_targets(self, idx_target, source, target, available_nodes, special_targets)

    ############################################################
    # Calculations
    ############################################################

    def infer_triadic_closure(self) -> float:
        """
        Approximates analytically the triadic closure value of the graph.

        Returns
        -------
        float
            triadic closure probability of the graph
        """
        tc = infer_triadic_closure(self)
        return tc

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
