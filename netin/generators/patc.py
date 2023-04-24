from typing import Union, Set, Tuple

import numpy as np
import networkx as nx

from netin.utils import constants as const
from netin.generators.tc import TriadicClosure
from .pa import PA


class PATC(PA, TriadicClosure):

    ############################################################
    # Constructor
    ############################################################

    def __init__(self, n: int, k: int, f_m: float, tc: float, seed: object = None):
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
        The initialization is an undirected with n nodes and no edges.
        Then, everytime a node is selected as source, it gets connected to k target nodes.
        Target nodes are selected via preferential attachment `in-degree` [BarabasiAlbert1999]_, or
        triadic closure `tc` [HolmeKim2002]_.
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
        Returns the initial set of special targets based on the triadic closure mechanism.

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

    def get_target_probabilities(self, source: Union[None, int], target_set: Union[None, Set[int]],
                                 special_targets: Union[None, object, iter] = None) -> tuple[np.array, set[int]]:
        """
        Returns the probabilities of selecting a target node from a set of nodes based on the preferential attachment
        or triadic closure.

        Parameters
        ----------
        source: int
            source node

        target_set: set[int]
            set of target nodes

        special_targets: object
            special targets

        Returns
        -------
        tuple[np.array, set[int]]
            probabilities of selecting a target node from a set of nodes, and the set of target nodes

        See Also
        --------
        :py:meth:`get_target_probabilities() <TriadicClosure.get_target_probabilities>` in :class:`netin.TC`.
        """
        return TriadicClosure.get_target_probabilities(self, source, target_set, special_targets)

    def get_target_probabilities_regular(self, source: Union[None, int],
                                         target_set: Union[None, Set[int]],
                                         special_targets: Union[None, object, iter] = None) -> \
            Tuple[np.array, Set[int]]:
        """
        Returns the probabilities of selecting a target node from a set of nodes based on the preferential attachment.

        Parameters
        ----------
        source: int
            source node

        target_set: set[int]
            set of target nodes

        special_targets: object
            special targets

        Returns
        -------
        tuple[np.array, set[int]]
            probabilities of selecting a target node from a set of nodes, and the set of target nodes

        See Also
        --------
        :py:meth:`get_target_probabilities() <PA.get_target_probabilities>` in :class:`netin.PA`.
        """
        return PA.get_target_probabilities(self, source, target_set, special_targets)

    def update_special_targets(self, idx_target: int, source: int, target: int, targets: Set[int],
                               special_targets: object) -> object:
        """
        Updates the set of special targets based on the triadic closure mechanism.

        Parameters
        ----------
        idx_target: int
            index of the target node

        source: int
            source node

        target: int
            target node

        targets: Set[int]
            set of target nodes

        special_targets: object
            special targets

        Returns
        -------
        object
            updated special targets

        See Also
        --------
        :func:`netin.TC.update_special_targets`

        """
        return TriadicClosure.update_special_targets(self, idx_target, source, target, targets, special_targets)

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
        tc = nx.average_clustering(self)
        return tc
