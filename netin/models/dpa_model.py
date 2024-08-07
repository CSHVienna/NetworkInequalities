from typing import Optional

import numpy as np

from netin.graphs.directed import DiGraph
from .directed_model import DirectedModel
from ..link_formation_mechanisms.indegree_preferential_attachment import InDegreePreferentialAttachment

class DPAModel(DirectedModel):
    pa: InDegreePreferentialAttachment

    def __init__(
            self, N: int, f: float, d: float,
            plo_M: float, plo_m: float,
            graph: Optional[DiGraph] = None,
            seed: int = 1):
        super().__init__(N, f, d, plo_M, plo_m, graph, seed)
        self.pa = InDegreePreferentialAttachment(graph=self.graph, n=N)

    def compute_target_probabilities(self, source: int) -> np.ndarray:
        return\
            super().compute_target_probabilities(source)\
            * self.pa.get_target_probabilities(source)
