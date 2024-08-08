from typing import Tuple

import numpy as np

from ..graphs.node_attributes import NodeAttributes
from .homophily import Homophily

class TwoClassHomophily(Homophily):
    @classmethod
    def from_two_class_homophily(
            cls, homophily: Tuple[float, float],
            node_class_values: NodeAttributes)\
            -> "TwoClassHomophily":

        return cls(
            node_class_values=node_class_values,
            homophily=np.asarray(
                [[homophily[0], 1 - homophily[0]],
                 [1 - homophily[1], homophily[1]]]),
            n_class_values=2
        )
