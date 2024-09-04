from typing import Union

import numpy as np

from .model import Model
from ..utils.constants import CLASS_ATTRIBUTE
from ..graphs.binary_class_node_vector import BinaryClassNodeVector
from ..graphs.binary_class_graph import\
    BinaryClassGraph, BinaryClassDiGraph

class BinaryClassModel(Model):
    f_m: float
    graph: Union[BinaryClassGraph, BinaryClassDiGraph]

    def __init__(
            self, *args,
            N: int, f_m: float,
            seed: Union[int, np.random.Generator] = 1,
            **kwargs):
        super().__init__(*args, N=N, seed=seed, **kwargs)
        self.f_m = f_m

    def _initialize_node_classes(self):
        if self.graph.has_node_class(CLASS_ATTRIBUTE):
            node_class_values_pre = self.graph.get_node_class(CLASS_ATTRIBUTE)
            assert isinstance(node_class_values_pre, BinaryClassNodeVector),\
            "The node class values must be binary"
            if len(node_class_values_pre) < self._n_nodes_total:
                ncv_post = BinaryClassNodeVector.from_fraction(
                    N=self._n_nodes_total,
                    f_m=self.f_m,
                    rng=self._rng)
                ncv_post[:len(node_class_values_pre)] =\
                node_class_values_pre.values
                self.graph.set_node_class(CLASS_ATTRIBUTE, ncv_post)
        elif isinstance(self.graph, (BinaryClassGraph, BinaryClassDiGraph)):
            self.graph.initialize_node_classes(
                N=self._n_nodes_total,
                f_m=self.f_m,
                rng=self._rng)
        else:
            raise ValueError(
                f"The graph must be a BinaryClassGraph or a BinaryClassDiGraph or have node classes `{CLASS_ATTRIBUTE}`")
