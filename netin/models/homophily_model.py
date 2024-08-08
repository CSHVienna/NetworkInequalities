from typing import Optional
from netin.graphs import Graph

import numpy as np

from ..link_formation_mechanisms.two_class_homophily import TwoClassHomophily
from .undirected_model import UndirectedModel

class HomophilyModel(UndirectedModel):
    def __init__(
            self, *args,
            N: int, m: int, f: float,
            h_m: float, h_M: float,
            graph: Optional[Graph] = None,
            **kwargs):
        super().__init__(*args, N=N, m=m, f=f, graph=graph, **kwargs)
        self.h = TwoClassHomophily.from_two_class_homophily(
            node_class_values=self.node_minority_class,
            homophily=(h_m, h_M))

    def compute_target_probabilities(self, source: int) -> np.ndarray:
        return super().compute_target_probabilities(source) * \
            self.h.get_target_probabilities(source)
