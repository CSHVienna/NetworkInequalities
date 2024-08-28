from typing import Union

import numpy as np

from .model import Model
from ..utils.constants import CLASS_ATTRIBUTE
from ..graphs.binary_class_node_vector import BinaryClassNodeVector

class BinaryClassModel(Model):
    f_m: float

    def __init__(
            self, *args,
            N: int, f_m: float,
            seed: Union[int, np.random.Generator] = 1,
            **kwargs):
        super().__init__(*args, N=N, seed=seed, **kwargs)
        self.f_m = f_m

    def _initialize_node_classes(self):
        node_class_values = BinaryClassNodeVector.from_fraction(
            N=self._n_nodes_total,
            f_m=self.f_m,
            rng=self._rng)
        if self.graph.has_node_class(CLASS_ATTRIBUTE):
            node_class_values_pre = self.graph.get_node_class(CLASS_ATTRIBUTE)
            assert isinstance(node_class_values_pre, BinaryClassNodeVector),\
            "The node class values must be binary"
            node_class_values[:len(node_class_values_pre)] =\
                node_class_values_pre.values
        self.graph.set_node_class(
            CLASS_ATTRIBUTE, node_class_values)
