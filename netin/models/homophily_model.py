from typing import Optional
from netin.graphs import Graph

import numpy as np

from ..link_formation_mechanisms.homophily import Homophily
from .undirected_model import UndirectedModel

class HomophilyModel(UndirectedModel):
    def __init__(self, N: int, m: int, f: float, h: float, graph: Optional[Graph] = None):
        super().__init__(N, m, f, graph)
        self.h = Homophily(
            node_class_values=self.node_minority_class,
            homophily=h)

    def compute_target_probabilities(self, source: int) -> np.ndarray:
        return super().compute_target_probabilities(source) * \
            self.h.get_target_probabilities(source)
