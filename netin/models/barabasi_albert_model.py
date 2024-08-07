from typing import Optional

import numpy as np

from netin.graphs import Graph
from .undirected_model import UndirectedModel
from ..link_formation_mechanisms.preferential_attachment import PreferentialAttachment

class BarabasiAlbertModel(UndirectedModel):
    def __init__(
            self, *args,
            N: int, m: int, f: float,
            graph: Optional[Graph] = None,
            **kwargs):
        super().__init__(
            *args, N=N, m=m, f=f, graph=graph, **kwargs)
        self.pa = PreferentialAttachment(n=N, graph=self.graph)

    def compute_target_probabilities(self, source: int) -> np.ndarray:
        return\
            super().compute_target_probabilities(source)\
            * self.pa.get_target_probabilities(source)
