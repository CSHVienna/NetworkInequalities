from typing import Optional

import numpy as np

from netin.graphs.directed import DiGraph
from .dpa_model import DPAModel
from ..link_formation_mechanisms.two_class_homophily import TwoClassHomophily

class DPAHModel(DPAModel):
    h: TwoClassHomophily

    def __init__(
            self, *args,
            N: int, f: float, d: float,
            plo_M: float, plo_m: float,
            h_m: float, h_M: float,
            graph: Optional[DiGraph] = None,
            seed: int = 1,
            **kwargs):
        super().__init__(
            *args, N=N, f=f, d=d,
            plo_M=plo_M, plo_m=plo_m,
            graph=graph, seed=seed,
            **kwargs)
        self.h = TwoClassHomophily.from_two_class_homophily(
            node_class_values=self.node_minority_class,
            homophily=(h_m, h_M))

    def compute_target_probabilities(self, source: int) -> np.ndarray:
        return\
            super().compute_target_probabilities(source)\
            * self.h.get_target_probabilities(source)
