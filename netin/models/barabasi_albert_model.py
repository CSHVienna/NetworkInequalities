from typing import Union

import numpy as np

from netin.graphs.graph import Graph

from .undirected_model import UndirectedModel
from ..link_formation_mechanisms.preferential_attachment import PreferentialAttachment

class BarabasiAlbertModel(UndirectedModel):
    def __init__(
            self, *args,
            N: int,
            m:int,
            seed:  Union[int, np.random.Generator] = 1,
            **kwargs):
        super().__init__(
            *args, N=N, m=m, seed=seed, **kwargs)
        self.pa = PreferentialAttachment(N=N, graph=self.graph)

    def compute_target_probabilities(self, source: int) -> np.ndarray:
        return\
            super().compute_target_probabilities(source)\
            * self.pa.get_target_probabilities(source)
