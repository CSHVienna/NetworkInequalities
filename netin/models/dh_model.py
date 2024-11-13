from typing import Union

import numpy as np

from .directed_model import DirectedModel
from ..utils.constants import CLASS_ATTRIBUTE
from ..link_formation_mechanisms.two_class_homophily import TwoClassHomophily

class DHModel(DirectedModel):
    """The DHmodel is a directed model that joins new nodes to
    the existing nodes with a probability proportional to the
    group assignment of the source and target nodes.

    Parameters
    ----------
    N : int
        Number of nodes to add.
    f_m : float
        The minority group fraction.
    d : float
        The target network density to reach.
    plo_M : float
        The power law exponent for the majority activity.
    plo_m : float
        The power law exponent for the minority activity.
    h_mm : float
        The homophily of the minority nodes.
    h_M : float
        The homophily of the majority nodes.
    seed : Union[int, np.random.Generator, None], optional
        Randomization seed or generator, by default None.
        If None, each run will be different.
    """
    SHORT = "DH"

    h_mm: float
    h_M: float

    h: TwoClassHomophily

    def __init__(
            self, *args,
            N: int, f_m: float, d: float,
            plo_M: float, plo_m: float,
            h_mm: float, h_M: float,
            seed: Union[int, np.random.Generator, None] = None, **kwargs):
        super().__init__(*args, N=N, f_m=f_m, d=d, plo_M=plo_M, plo_m=plo_m, seed=seed, **kwargs)
        self.h_mm = h_mm
        self.h_M = h_M

    def _initialize_lfms(self):
        """Initializes the link formation mechanisms for the DHModel.
        """
        super()._initialize_lfms()
        self.h = TwoClassHomophily.from_two_class_homophily(
            node_class_values=self.graph.get_node_class(CLASS_ATTRIBUTE),
            homophily=(self.h_M, self.h_mm))

    def compute_target_probabilities(self, source: int)\
        -> np.ndarray:
        """Computes the target probabilities based on :class:`TwoClassHomophily`.

        Parameters
        ----------
        source : int
            The source node.

        Returns
        -------
        np.ndarray
            The target node probabilities.
        """
        return super().compute_target_probabilities(source)\
            * self.h.get_target_probabilities(source)
