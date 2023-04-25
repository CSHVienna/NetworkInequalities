############################################
# System dependencies
############################################
import operator
from typing import List
from typing import Tuple

import numpy as np

import netin
from . import constants as const
from .sampling import Sampling


############################################
# Class
############################################
class PartialCrawls(Sampling):
    """Sampling by partial crawls.

    Parameters
    ----------
    g: netin.Graph | netin.DiGraph
        global network

    pseeds: float
        fraction of seeds to sample

    max_tries: int
        maximum number of tries to sample a subgraph with enough classes and edges

    random_seed: object
        seed for random number generator

    kwargs: dict
        additional parameters for the sampling method

    References
    ----------
    .. [Yang2017] J. Yang, B. Ribeiro, & J. Neville "Should We Be Confident in Peer Effects Estimated From Social Network Crawls?" ICWSM (Vol. 11, No. 1, pp. 708-711), 2017.
    """

    ######################################################
    # Constructor
    ######################################################
    def __init__(self, g: netin.Graph, pseeds: float, max_tries: int = const.MAX_TRIES,
                 random_seed: object = None, **kwargs):
        super().__init__(g, pseeds, max_tries, random_seed, **kwargs)
        self.snsize = kwargs.get('snsize', const.SNSIZE)

    @property
    def method_name(self) -> str:
        return const.PARTIAL_CRAWLS

    def sampling(self):
        super().sampling()

    def _sample(self):
        """
        Creates a subgraph from G based on random edge sampling
        """
        num_classes = 0

        ### 1. pick random neighbors including source node both at random
        while num_classes < const.MIN_CLASSES:
            super_node = self._get_super_node()
            edges = self._get_edges_from_tours(super_node)
            nodes = set(np.array(list(edges)).flatten().tolist())
            num_classes = self._count_classes(nodes)

        return nodes, edges

    def _get_super_node(self) -> List[int]:
        """
        Randomly selects (sn * N) nodes from the graph `g`.

        Returns
        -------
        list
            list of nodes
        """
        nodes = list(self.g.nodes())
        np.random.shuffle(nodes)
        super_node_size = int(round(self.g.number_of_nodes() * self.snsize))
        sn = nodes[:super_node_size]
        return sn

    def _get_edges_from_tours(self, super_node: List[int]) -> List[Tuple[int, int]]:
        """
        Sorts nodes in super node proportional to their edges outside the super node.
        Performs random walks from/to super node until collecting (pseeds * N) nodes
        returns a list od edges

        Parameters
        ----------
        super_node: list
            list of nodes in the super node

        Returns
        -------
        list
            list of edges
        """
        # proportional to the number of edges out of the super node
        sorted_S = {ni: len([nj for nj in self.g.neighbors(ni) if nj not in super_node]) for ni in self.g.nodes()}
        sorted_S = sorted(sorted_S.items(), key=operator.itemgetter(1), reverse=True)
        sorted_S = [n[0] for n in sorted_S]

        # tours
        edges = set()
        sampled_nodes = set()
        for vi in sorted_S:
            nbrs_i = list(self.g.neighbors(vi))
            if len(nbrs_i) == 0:
                continue
            vj = np.random.choice(nbrs_i, 1)[0]  # random neighbor
            while vj not in super_node and len(sampled_nodes) < self.nseeds:
                edges.add((vi, vj))
                sampled_nodes |= {vi, vj}
                vi = vj
                nbrs_i = list(self.g.neighbors(vi))
                if len(nbrs_i) == 0:
                    break
                vj = np.random.choice(nbrs_i, 1)[0]  # random neighbor

        return edges
