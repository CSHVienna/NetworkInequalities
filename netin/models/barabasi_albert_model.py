from typing import Union

import numpy as np

from .undirected_model import UndirectedModel
from ..link_formation_mechanisms.preferential_attachment import PreferentialAttachment

class BarabasiAlbertModel(UndirectedModel):
    pa: PreferentialAttachment

    def compute_target_probabilities(self, source: int) -> np.ndarray:
        return\
            super().compute_target_probabilities(source)\
            * self.pa.get_target_probabilities(source)

    def _initialize_lfms(self):
        self.pa = PreferentialAttachment(
            N=self._n_nodes_total, graph=self.graph)
