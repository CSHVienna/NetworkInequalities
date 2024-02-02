from typing import Optional
from netin.graphs import Graph

import numpy as np

from ..link_formation_mechanisms.homophily import Homophily
from .model import Model

class HomophilyModel(Model):
    def __init__(self, N: int, m: int, f: float, h: float, graph: Optional[Graph] = None):
        super().__init__(N, m, f, graph)
        self.h = Homophily(self.node_class_values, h)

    def compute_target_probabilities(self, source: int) -> np.ndarray:
        return self.h.get_target_probabilities(source)
