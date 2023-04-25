############################################
# System dependencies
############################################
import numpy as np

import netin
from . import constants as const
from .sampling import Sampling


############################################
# Class
############################################
class RandomNeighbor(Sampling):
    """Random neighbor sampling.

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
    """

    ######################################################
    # Constructor
    ######################################################
    def __init__(self, g: netin.Graph, pseeds: float, max_tries: int = const.MAX_TRIES,
                 random_seed: object = None):
        super().__init__(g, pseeds, max_tries, random_seed)

    @property
    def method_name(self) -> str:
        return const.RANDOM_NEIGHBORS

    def sampling(self):
        super().sampling()

    def _sample(self):
        """
        Creates a subgraph from G based on random edge sampling
        """
        num_classes = 0
        nodes = set()
        edges = set()

        ### 1. pick random neighbors including source node both at random
        while num_classes < const.MIN_CLASSES:
            tmp_nodes = list(self.g.nodes())
            np.random.shuffle(tmp_nodes)

            for node in tmp_nodes:
                tmp_edges = list(self.g.edges(node))
                if len(tmp_edges) == 0:
                    continue
                np.random.shuffle(tmp_edges)
                edge = tmp_edges[0]
                nodes |= set(edge)
                edges |= set([edge])

                if len(nodes) >= self.nseeds:
                    break

            num_classes = self._count_classes(nodes)

        return nodes, edges
