from typing import Optional, Union

import numpy as np

from netin.graphs.directed import DiGraph
from .directed_model import DirectedModel
from ..link_formation_mechanisms.indegree_preferential_attachment import InDegreePreferentialAttachment

class DPAModel(DirectedModel):
    pa: InDegreePreferentialAttachment

    def __init__(
            self, *args,
            N: int, f_m: float, d: float,
            plo_M: float, plo_m: float,
            graph: Optional[DiGraph] = None,
            seed:  Union[int, np.random.Generator] = 1,
            **kwargs):
        super().__init__(
            *args, N=N, f_m=f_m, d=d,
            plo_M=plo_M, plo_m=plo_m,
            graph=graph, seed=seed,
            **kwargs)

    def compute_target_probabilities(self, source: int) -> np.ndarray:
        return\
            super().compute_target_probabilities(source)\
            * self.pa.get_target_probabilities(source)
