from typing import Tuple, Union

import numpy as np

from ..graphs.node_vector import NodeVector
from ..utils.validator import validate_float
from .homophily import Homophily

class TwoClassHomophily(Homophily):
    def __init__(
            self,
            node_class_values: NodeVector,
            homophily: Union[float, np.ndarray], n_class_values: Union[int, None] = None) -> None:
        super().__init__(node_class_values, homophily, n_class_values)
        assert self.h.shape == (2, 2),\
            ("Homophily matrix must be 2x2. Use "
             "`Homophily` to define homophily for more than two classes.")

    @classmethod
    def from_two_class_homophily(
            cls, homophily: Tuple[float, float],
            node_class_values: NodeVector)\
            -> "TwoClassHomophily":

        assert(len(homophily) == 2),\
            ("Homophily for two classes must be a tuple of two values. "
             f"Received {homophily}.")
        for h in homophily:
            validate_float(h, minimum=0., maximum=1.)
        return cls(
            node_class_values=node_class_values,
            homophily=np.asarray(
                [[homophily[0], 1 - homophily[0]],
                 [1 - homophily[1], homophily[1]]]),
            n_class_values=2
        )
