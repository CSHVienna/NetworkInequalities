import numpy as np

from .undirected_model import UndirectedModel
from ..link_formation_mechanisms.preferential_attachment import PreferentialAttachment

class BarabasiAlbertModel(UndirectedModel):
    """The BarabasiAlbertModel join new nodes to the existing nodes with a
    probability proportional to the degree of the existing nodes.

    References
    ----------
    .. [BarabasiAlbert1999] A. L. Barabasi and R. Albert "Emergence of scaling in random networks", Science 286, pp 509-512, 1999.
    """
    SHORT = "BA"
    pa: PreferentialAttachment

    def compute_target_probabilities(self, source: int) -> np.ndarray:
        """Compute the target probabilities for the BarabasiAlbertModel.
        Connection probability is proportional to the degree of the existing nodes.

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
        self.pa = PreferentialAttachment(
            N=self._n_nodes_total, graph=self.graph)

    def _initialize_node_classes(self):
        pass # Not needed for this model
