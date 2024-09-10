from typing import Any, Dict, Optional, Union

import numpy as np

from ..graphs import Graph
from ..utils.validator import validate_int
from .model import Model

class UndirectedModel(Model):
    """The base class for all undirected models.
    Based on [BarabasiAlbert1999]_,
    this model grows a network by adding a total of :attr:`N` nodes to the network.
    Each node that is added connects to the previously added nodes with :attr:`m` links.
    How the target nodes are chosen depends on link formation mechanisms.
    The implementation of these mechanisms is handled
    by subclasses (see for instance, :class:`.PAHModel`).

    Parameters
    ----------
    N : int
        Number of nodes to be added.
    m : int
        Number of links for each new node.
    seed : Union[int, np.random.Generator], optional
        The randomization seed or random number generator, by default 1
    """

    SHORT = "UNDIRECTED"

    m: int

    def __init__(
            self, *args,
            N: int,
            m:int,
            seed: Optional[Union[int, np.random.Generator]] = None,
            **kwargs):
        validate_int(N, minimum=1)
        validate_int(m, minimum=1)
        self.m = m
        super().__init__(
            *args,
            N=N,
            seed=seed,
            **kwargs)

    def _initialize_empty_graph(self) -> Graph:
        return Graph()

    def _populate_initial_graph(self) -> Graph:
        for i in range(self.m):
            self.graph.add_node(i)
        return self.graph

    def _simulate(self) -> Graph:
        """Simulates the undirected model.
        After adding :attr:`m` initial nodes, ``N-m`` nodes are added one after the other.
        Each new node connects to previously added nodes with :attr:`m` links.
        The choice of target nodes depends on the implementation of
        :class:`.LinkFormationMechanism` and :class:`.Filter`.
        This should be implemented in respective subclasses.

        Returns
        -------
        Graph
            The simulated network.
        """
        n_nodes = len(self.graph)
        for source in range(
                n_nodes, self._n_nodes_total):
            self.graph.add_node(source)
            for _ in range(self.m):
                target_probabilities = self.compute_target_probabilities(source)[:source]
                target_probabilities /= target_probabilities.sum()
                target = self._sample_target_node(target_probabilities)
                self.graph.add_edge(source, target)
        return self.graph

    def get_metadata(
            self,
            d_meta_data: Optional[Dict[str, Any]] = None)\
                -> Dict[str, Any]:
        """Adds the number of links per new node :attr:`m` to the metadata dictionary.

        Returns
        -------
        Dict[str, Any]
            The (updated or created) metadata dictionary.
        """
        d = super().get_metadata(d_meta_data)
        d[self.__class__.__name__] = {
            "m": self.m,
        }
        return d

    def preload_graph(self, graph: Graph):
        """Preloads a graph.

        Parameters
        ----------
        graph : Graph
            Graph to be preloaded.
        """
        assert not graph.is_directed(),\
            "The graph must not be directed."
        return super().preload_graph(graph)
