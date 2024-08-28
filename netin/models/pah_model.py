import numpy as np

from .homophily_model import HomophilyModel
from ..link_formation_mechanisms.preferential_attachment import PreferentialAttachment

class PAHModel(HomophilyModel):
    pa: PreferentialAttachment

    def compute_target_probabilities(self, source: int) -> np.ndarray:
        """Computes the target probabilities for the source node.
        The probability to connect to a target node is computed as the
        product of the probabilities of the preferential attachment
        and homophily mechanisms.

        Parameters
        ----------
        source : int
            Source node.

        Returns
        -------
        np.ndarray
            Target probabilities.
        """
        p_target =\
            super().compute_target_probabilities(source)\
            * self.pa.get_target_probabilities(source)\
            * self.h.get_target_probabilities(source)
        return p_target / p_target.sum()

    def _initialize_lfms(self):
        """Initializes the link formation mechanisms.
        """
        super()._initialize_lfms()
        self.pa = PreferentialAttachment(
            N=self._n_nodes_total,
            graph=self.graph)
