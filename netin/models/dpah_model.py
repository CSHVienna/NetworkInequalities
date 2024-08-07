from typing import Optional

import numpy as np

from netin.graphs.directed import DiGraph
from .dpa_model import DPAModel
from ..link_formation_mechanisms.homophily import Homophily

class DPAHModel(DPAModel):
    h: Homophily

    def __init__(
            self, N: int, f: float, d: float,
            plo_M: float, plo_m: float,
            h: float,
            graph: Optional[DiGraph] = None,
            seed: int = 1):
        super().__init__(N, f, d, plo_M, plo_m, graph, seed)
        self.h = Homophily(
            node_class_values=self.node_minority_class,
            homophily=h)

    def compute_target_probabilities(self, source: int) -> np.ndarray:
        return\
            super().compute_target_probabilities(source)\
            * self.h.get_target_probabilities(source)
