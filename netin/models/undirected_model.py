from typing import Any, Dict, Optional, Union

import numpy as np

from ..graphs import Graph
from ..utils.validator import validate_int
from .model import Model

class UndirectedModel(Model):
    m: int

    def __init__(
            self, *args,
            N: int,
            m:int,
            seed: Union[int, np.random.Generator] = 1,
            **kwargs):
        validate_int(N, minimum=1)
        validate_int(m, minimum=1)
        self.m = m
        super().__init__(
            *args,
            N=N,
            seed=seed,
            **kwargs)

    def _initialize_graph(self):
        self.graph = Graph()

    def _populate_initial_graph(self):
        for i in range(self.m):
            self.graph.add_node(i)
            for j in range(i):
                self.graph.add_edge(i, j)

    def _simulate(self) -> Graph:
        n_nodes = len(self.graph)
        for source in range(
                n_nodes, n_nodes + self.N):
            self.graph.add_node(source)
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
