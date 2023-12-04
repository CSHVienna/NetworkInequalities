from abc import abstractmethod
from typing import Union, Dict, Any, List, Tuple

import numpy as np

from netin.generators.undirected import UnDiGraph
from netin.generators.tc import TriadicClosure
from netin.utils import constants as const


class GraphTC(UnDiGraph, TriadicClosure):
    """Abstract base class for undirected triadic closure graphs.

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

    tc_uniform: bool
        specifies whether the triadic closure target is chosen uniform at random or if it follows the regular link formation mechanisms (e.g., homophily) (default=True)

    Notes
    -----
    The initialization is an undirected graph with n nodes and no edges.
    The first `k` nodes are marked as active.
    Then, every node, starting from `k+1` is selected as source in order of their id.
    When selected as source, a node connects to `k` active target nodes which were chosen as sources before.
    Target nodes are selected via any link formation mechanism (to be implemented in sub-classes) with probability ``1-p_{TC}``,
    and with probability ``p_{TC}`` via triadic closure (see :class:`netin.TriadicClosure`) [HolmeKim2002]_.
    Afterwards, the source node is flagged as being active (so that subsequent sources can connect to it).

    Note that this model is still work in progress and not fully implemented yet.
    """

    ############################################################
    # Constructor
    ############################################################

    def __init__(self, n: int, k: int, f_m: float, tc: float, tc_uniform: bool = True,
                 seed: object = None):
        UnDiGraph.__init__(self, n, k, f_m, seed)
        TriadicClosure.__init__(self, n=n, f_m=f_m, tc=tc, seed=seed)
        self.tc_uniform = tc_uniform
        self.model_name = const.TCH_MODEL_NAME

    def get_metadata_as_dict(self) -> Dict[str, Any]:
        """
        Returns the metadata (parameters) of the model as a dictionary.

        Returns
        -------
        dict
            metadata of the model
        """
        obj = UnDiGraph.get_metadata_as_dict(self)
        obj.update({
            'tc_uniform': self.tc_uniform
        })
        return obj

    ############################################################
    # Generation
    ############################################################

    def get_target_probabilities(self, source: int, available_nodes: List[int]) -> Tuple[np.array, List[int]]:
        """
        Returns the probabilities of nodes to be selected as target nodes.

        Parameters
        ----------
        source: int
            source node id

        available_nodes: List[int]
            list of available nodes to connect to

        Returns
        -------
        Tuple[np.ndarray, List[int]]
            probabilities of nodes to be selected as target nodes, and list of target of nodes

        """
        tc_prob = np.random.random()

        if source != self._node_source_curr:
            self.init_special_targets(source)

        if tc_prob < self.tc and len(self._tc_candidates) > 0:
            if not self.tc_uniform:
                # Triadic closure is uniform
                return self.get_target_probabilities_regular(
                    source,
                    list(self._tc_candidates.keys()))
            return TriadicClosure\
                .get_target_probabilities(self, source, available_nodes)

        # Edge is added based on regular mechanism (not triadic closure)
        return self.get_target_probabilities_regular(source, available_nodes)

    @abstractmethod
    def get_target_probabilities_regular(self, source: int, target_list: List[int]) -> \
            Tuple[np.ndarray, List[int]]:
        raise NotImplementedError

    ############################################################
    # Calculations
    ############################################################

    def info_params(self):
        """
        Shows the (input) parameters of the graph.
        """
        print('tc_uniform: {}'.format(self.tc_uniform))
        TriadicClosure.info_params(self)

    def info_computed(self):
        """
        Shows the (computed) properties of the graph.
        """
        TriadicClosure.info_computed(self)


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
                              seed=self.seed)
