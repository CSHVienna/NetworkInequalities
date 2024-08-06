from typing import Dict, Optional, Any
from abc import ABC, abstractmethod

import numpy as np

from ..graphs.graph import Graph
from ..graphs.node_attributes import NodeAttributes
from ..base_class import BaseClass

class Model(ABC, BaseClass):
    """Model class.
    Abstract class that defines a growing network model.
    Specific growing-network-model implementations should inherit from this class and implement the provided abstract methods.
    """
    N: int
    m: int
    f: float
    graph: Graph
    node_minority_class: NodeAttributes
    seed: int

    def __init__(
            self, N: int, m: int, f: float,
            graph: Optional[Graph] = None,
            seed: int = 1):
        """Creates a new instance of the Model class.

        Parameters
        ----------
        N : int
            Number of final nodes in the network.
        m : int
            Number of links that each new node creates.
        f : float
            Fraction of nodes that belong to the minority class.
        graph : Optional[Graph], optional
            If present, an existing network that will be extended. In this case, `N >= graph.number_of_nodes()` as the graph will be extended by the missing nodes. If no graph is given, the model creates its own graph and initializes it with `m` fully connected nodes.
            Calling the `simulate`-function will then add the remaining `N - m` nodes.
        seed : int, optional
            A random seed for reproducibility, by default 1
        """
        super().__init__()

        self.N = N
        self.m = m
        self.f = f
        self.seed = seed
        np.random.seed(seed)

        self.node_minority_class = NodeAttributes.from_ndarray(
            np.where(np.random.rand(N) < f, 1, 0), name="minority_class")

        if graph is None:
            self.graph = Graph()
        else:
            self.graph = graph
            self._initialize_graph()
        self._initialize_lfms()

    def _initialize_graph(self):
        for i in range(self.m):
            self.graph.add_node(i)
            for j in range(i):
                self.graph.add_edge(i, j)

    @abstractmethod
    def _initialize_lfms(self):
        pass

    @abstractmethod
    def compute_target_probabilities(self, source: int) -> np.ndarray:
        pass

    def simulate(self) -> Graph:
        for source in range(
                self.graph.number_of_nodes(), self.N):
            self.graph.add_node(
                source,
                minority=self.node_minority_class[source])
            for _ in range(self.m):
                target_probabilities = self.compute_target_probabilities(source)[:source]
                target_probabilities /= target_probabilities.sum()
                target = np.random.choice(
                    np.arange(source),
                    p=target_probabilities[:source])
                self.graph.add_edge(source, target)
        return self.graph

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(N={self.N}, m={self.m}, f={self.f})"

    def get_metadata(self, d_meta_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        d = super().get_metadata(d_meta_data)
        d[self.__class__.__name__] = {
            "N": self.N,
            "m": self.m,
            "f": self.f,
            "seed": self.seed
        }
        self.graph.get_metadata(
            d[self.__class__.__name__])
        self.node_minority_class.get_metadata(
            d[self.__class__.__name__])

        return d
