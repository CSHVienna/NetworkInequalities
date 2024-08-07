from typing import Optional

import numpy as np

from ..graphs import Graph
from .barabasi_albert_model import BarabasiAlbertModel
from .homophily_model import HomophilyModel

class PAHModel(BarabasiAlbertModel, HomophilyModel):
    def __init__(
            self, N: int, m: int, f: float, h: float,
            graph: Optional[Graph] = None):
        BarabasiAlbertModel.__init__(self, N, m, f, graph)
        graph = self.graph # Needed to not initialize the graph twice
        HomophilyModel.__init__(self, N, m, f, h, graph)

    def compute_target_probabilities(self, source: int) -> np.ndarray:
        """Computes the target probabilities for the source node.
        The probability to connect to a target node is computed as the product of the probabilities of the preferential attachment and homophily mechanisms.

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
