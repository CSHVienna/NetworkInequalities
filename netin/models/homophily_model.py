from typing import Union

import numpy as np

from ..utils.constants import CLASS_ATTRIBUTE
from ..link_formation_mechanisms.two_class_homophily import TwoClassHomophily
from .undirected_model import UndirectedModel
from .binary_class_model import BinaryClassModel

class HomophilyModel(UndirectedModel, BinaryClassModel):
    h_m: float
    h_M: float

    h: TwoClassHomophily

    def __init__(
            self, *args,
            N: int, f_m: float, m:int,
            h_m: float, h_M: float,
            seed:  Union[int, np.random.Generator] = 1,
            **kwargs):
        super().__init__(
            *args, N=N, m=m, f_m=f_m,
            seed=seed, **kwargs)
        self.h_m = h_m
        self.h_M = h_M

    def compute_target_probabilities(self, source: int) -> np.ndarray:
        return super().compute_target_probabilities(source) * \
            self.h.get_target_probabilities(source)

    def _initialize_lfms(self):
        self.h = TwoClassHomophily.from_two_class_homophily(
            node_class_values=self.graph.get_node_class(CLASS_ATTRIBUTE),
            homophily=(self.h_m, self.h_M))
