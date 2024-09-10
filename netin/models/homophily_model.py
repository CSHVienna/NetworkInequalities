from typing import Union, Optional

import numpy as np

from ..utils.constants import CLASS_ATTRIBUTE
from ..link_formation_mechanisms.two_class_homophily import TwoClassHomophily
from .undirected_model import UndirectedModel
from .binary_class_model import BinaryClassModel

class HomophilyModel(UndirectedModel, BinaryClassModel):
    SHORT = "HOMOPHILY"
    h_m: float
    h_M: float

    h: TwoClassHomophily

    def __init__(
            self, *args,
            N: int, f_m: float, m:int,
            h_m: float, h_M: float,
            seed:  Optional[Union[int, np.random.Generator]] = None,
            **kwargs):
        """A simple homophily model which connects nodes with a probability proportional to homophily, the tendency of individuals to associate with others who are similar to themselves.

        Parameters
        ----------
        N : int
            Number of nodes.
        f_m : float
            Fraction of minority nodes.
        m : int
            Number of edges to attach from a new node to existing nodes.
        h_m : float
            Homophily of the minority nodes.
            If a minority node joins the network, it connects to other
            minority nodes with probability ``h_m`` and to the majority
              group with the complementary probability ``1-h_m``.
        h_M : float
            Homophily of the majority nodes.
            If a majority node joins the network, it connects to other
            majority nodes with probability ``h_M`` and to the minority
            group with the complementary probability ``1-h_M``.
        seed : Union[int, np.random.Generator], optional
            Randomization seed or random number generator, by default 1
        """
        super().__init__(
            *args, N=N, m=m, f_m=f_m,
            seed=seed, **kwargs)
        self.h_m = h_m
        self.h_M = h_M

    def compute_target_probabilities(self, source: int) -> np.ndarray:
        """Compute the target probabilities for the HomophilyModel
        based on the Homophily link formation mechanism."""
        return super().compute_target_probabilities(source) * \
            self.h.get_target_probabilities(source)

    def _initialize_lfms(self):
        self.h = TwoClassHomophily.from_two_class_homophily(
            node_class_values=self.graph.get_node_class(CLASS_ATTRIBUTE),
            homophily=(self.h_m, self.h_M))
