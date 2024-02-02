from typing import Optional

import numpy as np

from netin.graphs import Graph
from .model import Model
from ..link_formation_mechanisms.preferential_attachment import PreferentialAttachment

class BarabasiAlbertModel(Model):
    def __init__(self, N: int, m: int, f: float, graph: Optional[Graph] = None):
        super().__init__(N, m, f, graph)
        self.pa = PreferentialAttachment(n=N, graph=self.graph)

    def compute_target_probabilities(self, source: int) -> np.ndarray:
        return self.pa.get_target_probabilities(source)
