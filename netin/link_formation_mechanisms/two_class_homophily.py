from typing import Tuple, Union

import numpy as np

from ..graphs.binary_class_node_vector import BinaryClassNodeVector
from ..utils.validator import validate_float
from .homophily import Homophily

class TwoClassHomophily(Homophily):
    """Two class homophily link formation mechanism.

    This is a convenience class for the `Homophily` link formation mechanism with two classes.

    Parameters
    ----------
    node_class_values : CategoricalNodeVector
        The class assignment for each node (dimensions `n_nodes`).
    homophily : Union[float, np.ndarray]
        The homophily value(s).
    """
    def __init__(
            self,
            node_class_values: BinaryClassNodeVector,
            homophily: Union[float, np.ndarray]) -> None:
        super().__init__(
            node_class_values=node_class_values, homophily=homophily)
        assert node_class_values.n_values == 2,\
            ("TwoClassHomophily can only be used for two classes. "
             f"Received {node_class_values.n_values} classes")
        assert self.h.shape == (2, 2),\
            ("Homophily matrix must be 2x2. Use "
             "`Homophily` to define homophily for more than two classes.")

    @classmethod
    def from_two_class_homophily(
            cls, homophily: Tuple[float, float],
            node_class_values: BinaryClassNodeVector)\
            -> "TwoClassHomophily":
        """Creates a TwoClassHomophily instance from two homophily values.
        """
        assert(len(homophily) == 2),\
            ("Homophily for two classes must be a tuple of two values. "
             f"Received {homophily}.")
        for h in homophily:
            validate_float(h, minimum=0., maximum=1.)
        return cls(
            node_class_values=node_class_values,
            homophily=np.asarray(
                [[homophily[0], 1 - homophily[0]],
                 [1 - homophily[1], homophily[1]]]))
