from typing import Any, Dict, Optional

import numpy as np

from ..graphs import Graph
from .model import Model

class UndirectedModel(Model):
    def __init__(self, N: int, m: int, f: float, graph=None):
        super().__init__(N, f, graph)
        self.m = m

    def _initialize_graph(self):
        self.graph = Graph()

    def _populate_initial_graph(self):
        for i in range(self.m):
            self.graph.add_node(i)
            for j in range(i):
                self.graph.add_edge(i, j)

    def simulate(self) -> Graph:
        for source in range(
                self.graph.number_of_nodes(), self.N):
            self.graph.add_node(
                source,
                minority=self.node_minority_class[source])
            for _ in range(self.m):
                target_probabilities = self.compute_target_probabilities(source)[:source]
                target_probabilities /= target_probabilities.sum()
                target = self._sample_target_node(target_probabilities[:source])
                self.graph.add_edge(source, target)
        return self.graph

    def get_metadata(
            self,
            d_meta_data: Optional[Dict[str, Any]] = None)\
                -> Dict[str, Any]:
        d = super().get_metadata(d_meta_data)
        d[self.__class__.__name__] = {
            "m": self.m,
        }
        return d
