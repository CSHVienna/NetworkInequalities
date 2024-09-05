import numpy as np

from .directed_model import DirectedModel
from ..link_formation_mechanisms.indegree_preferential_attachment import InDegreePreferentialAttachment

class DPAModel(DirectedModel):
    SHORT = "DPA"
    pa: InDegreePreferentialAttachment

    def compute_target_probabilities(self, source: int) -> np.ndarray:
        return\
            super().compute_target_probabilities(source)\
            * self.pa.get_target_probabilities(source)

    def _initialize_lfms(self):
        super()._initialize_lfms()
        self.pa = InDegreePreferentialAttachment(
            graph=self.graph, N=self._n_nodes_total)
