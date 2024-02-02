from typing import Optional

import numpy as np

from ..graphs import Graph
from .barabasi_albert_model import BarabasiAlbertModel
from .homophily_model import HomophilyModel

class PAH(BarabasiAlbertModel, HomophilyModel):
    def __init__(self, N: int, m: int, f: float, h: float, graph: Optional[Graph] = None):
        BarabasiAlbertModel.__init__(self, N, m, f, graph)
        graph = self.graph
        HomophilyModel.__init__(self, N, m, f, h, graph)

    def compute_target_probabilities(self, source: int) -> np.ndarray:
        p_target = self.pa.get_target_probabilities(source) * self.h.get_target_probabilities(source)
        return p_target / p_target.sum()
