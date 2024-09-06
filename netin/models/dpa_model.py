import numpy as np

from .directed_model import DirectedModel
from ..link_formation_mechanisms.indegree_preferential_attachment import InDegreePreferentialAttachment

class DPAModel(DirectedModel):
    """The DPAModel is a directed model that joins new nodes to the existing nodes
    with a probability proportional to the in-degree of the existing nodes.

    For the simulation logic of this model, see the base `DirectedModel`.
    """
    SHORT = "DPA"
    pa: InDegreePreferentialAttachment

    def compute_target_probabilities(self, source: int) -> np.ndarray:
        """Compute the target probabilities for the DPAModel.
        Connection probability is proportional to the in-degree of the existing nodes.

        Parameters
        ----------
        source : int
            The source node.

        Returns
        -------
        np.ndarray
            The target probabilities.
        """
        return\
            super().compute_target_probabilities(source)\
            * self.pa.get_target_probabilities(source)

    def _initialize_lfms(self):
        super()._initialize_lfms()
        self.pa = InDegreePreferentialAttachment(
            graph=self.graph, N=self._n_nodes_total)
