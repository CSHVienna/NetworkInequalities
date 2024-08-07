from typing import Optional

import numpy as np

from .undirected_model import UndirectedModel
from ..graphs import Graph
from ..link_formation_mechanisms.homophily import Homophily
from ..link_formation_mechanisms.preferential_attachment import PreferentialAttachment

class PAHModel(UndirectedModel):
    def __init__(
            self, *args,
            N: int, m: int, f: float, h: float,
            graph: Optional[Graph] = None,
            **kwargs):
        super().__init__(
            *args, N=N, m=m, f=f, graph=graph, **kwargs)
        self.h = Homophily(
            node_class_values=self.node_minority_class, homophily=h)
        self.pa = PreferentialAttachment(n=N, graph=self.graph)

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
