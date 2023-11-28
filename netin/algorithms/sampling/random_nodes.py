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
class RandomNodes(Sampling):
    """Random node sampling.

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
                 random_seed: object = None, **kwargs):
        super().__init__(g, pseeds, max_tries, random_seed, **kwargs)

    @property
    def method_name(self) -> str:
        return const.RANDOM_NODES

    def sampling(self):
        super().sampling()

    def _sample(self):
        """
        Creates a subgraph from G based on random node sampling
        """
        num_classes = 0
        edges = None

        ### 1. pick random nodes
        while num_classes < const.MIN_CLASSES:
            nodes = list(self.g.node_list)
            np.random.shuffle(nodes)
            nodes = nodes[:self.nseeds]
            num_classes = self._count_classes(nodes)

        return nodes, edges
