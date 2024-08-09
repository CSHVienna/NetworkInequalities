from typing import Optional

import numpy as np

from .undirected_model import UndirectedModel
from ..graphs import Graph
from ..link_formation_mechanisms.two_class_homophily import TwoClassHomophily
from ..link_formation_mechanisms.preferential_attachment import PreferentialAttachment

class PAHModel(UndirectedModel):
    def __init__(
        self, *args,
        N: int, f: float,
        m:int,
        h_m: float, h_M: float,
        graph: Optional[Graph] = None,
        seed: int = 1,
        **kwargs):
        super().__init__(
            *args, N=N, m=m, f=f,
            graph=graph, seed=seed, **kwargs)
        self.h = TwoClassHomophily.from_two_class_homophily(
            node_class_values=self.node_minority_class,
            homophily=(h_m, h_M))
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
