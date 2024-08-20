from typing import Optional
from netin.graphs import Graph

import numpy as np

from ..utils.constants import CLASS_ATTRIBUTE
from ..graphs.binary_class_node_vector import BinaryClassNodeVector
from ..link_formation_mechanisms.two_class_homophily import TwoClassHomophily
from .undirected_model import UndirectedModel

class HomophilyModel(UndirectedModel):
    def __init__(
            self, *args,
            N: int, f: float,
            m:int,
            h_m: float, h_M: float,
            graph: Optional[Graph] = None,
            seed: int = 1,
            **kwargs):
        super().__init__(
            *args, N=N, m=m,
            node_attributes={
                CLASS_ATTRIBUTE:\
                    BinaryClassNodeVector.from_fraction(N=N, f_m=f)
            },
            graph=graph,
            seed=seed, **kwargs)
        self.h = TwoClassHomophily.from_two_class_homophily(
            node_class_values=self.node_attributes[CLASS_ATTRIBUTE],
            homophily=(h_m, h_M))

    def compute_target_probabilities(self, source: int) -> np.ndarray:
        return super().compute_target_probabilities(source) * \
            self.h.get_target_probabilities(source)
