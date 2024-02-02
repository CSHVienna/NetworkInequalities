from typing import Optional
from abc import ABC, abstractmethod

import numpy as np

from ..graphs.graph import Graph

class Model(ABC):
    def __init__(self, N: int, m: int, f: float, graph: Optional[Graph] = None):
        self.N = N
        self.m = m
        self.f = f

        self.node_class_values = np.where(np.random.rand(N) < f, 1, 0)

        if graph is None:
            self.graph = Graph(n=self.N, f_m=self.f)
            # @TODO: Remove the parameters
        else:
            self.graph = graph

    @abstractmethod
    def compute_target_probabilities(self, source: int) -> np.ndarray:
        pass

    def simulate(self) -> Graph:
        self._initialize_graph()

        for source in range(self.m, self.N):
            self.graph.add_node(source,
                                minority=self.node_class_values[source])
            for _ in range(self.m):
                target_probabilities = self.compute_target_probabilities(source)[:source]
                target_probabilities /= target_probabilities.sum()
                target = np.random.choice(
                    np.arange(source),
                    p=target_probabilities[:source])
                self.graph.add_edge(source, target)
        return self.graph

    def _initialize_graph(self):
        for i in range(self.m):
            self.graph.add_node(i)
            for j in range(i):
                self.graph.add_edge(i, j)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(N={self.N}, m={self.m}, f={self.f})"
