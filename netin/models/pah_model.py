import numpy as np

from .homophily_model import HomophilyModel
from ..link_formation_mechanisms.preferential_attachment import PreferentialAttachment

class PAHModel(HomophilyModel):
    """The PAHModel implements the [P]referential [A]ttachment and [H]omophily model.

    Nodes join the network by connecting to existing
    nodes proportional to their degree and group assignment.
    See :class:`.HomophilyModel` for how to parameterize the homophily values.

    This model is based on [Karimi2018]_.
    """
    SHORT = "PAH"
    pa: PreferentialAttachment

    def compute_target_probabilities(self, source: int)\
            -> np.ndarray:
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
            * self.pa.get_target_probabilities(source)
        return p_target / p_target.sum()

    def _initialize_lfms(self):
        """Initializes the link formation mechanisms.
        """
        super()._initialize_lfms()
        self.pa = PreferentialAttachment(
            N=self._n_nodes_total,
            graph=self.graph)
