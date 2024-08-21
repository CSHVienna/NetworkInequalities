from typing import Union

import numpy as np

from .dpa_model import DPAModel
from ..link_formation_mechanisms.two_class_homophily import TwoClassHomophily
from ..link_formation_mechanisms.indegree_preferential_attachment import InDegreePreferentialAttachment
from ..utils.constants import CLASS_ATTRIBUTE

class DPAHModel(DPAModel):
    h_m: float
    h_M: float

    h: TwoClassHomophily
    pa: InDegreePreferentialAttachment

    def __init__(
            self, *args,
            N: int, f_m: float, d: float,
            plo_M: float, plo_m: float,
            h_m: float, h_M: float,
            seed:  Union[int, np.random.Generator] = 1,
            **kwargs):
        super().__init__(
            *args, N=N, f_m=f_m, d=d,
            plo_M=plo_M, plo_m=plo_m,
            seed=seed,
            **kwargs)
        self.h_m = h_m
        self.h_M = h_M

    def _initialize_lfms(self):
        super()._initialize_lfms()
        self.h = TwoClassHomophily.from_two_class_homophily(
            node_class_values=self.graph.get_node_class(CLASS_ATTRIBUTE),
            homophily=(self.h_m, self.h_M))
        self.pa = InDegreePreferentialAttachment(
            graph=self.graph, N=self.N)

    def compute_target_probabilities(self, source: int) -> np.ndarray:
        return\
            super().compute_target_probabilities(source)\
            * self.h.get_target_probabilities(source)
