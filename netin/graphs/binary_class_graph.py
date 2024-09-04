import numpy as np

from .graph import Graph
from .directed import DiGraph
from .binary_class_node_vector import BinaryClassNodeVector
from ..utils.constants import CLASS_ATTRIBUTE

class BinaryClassGraphMixin:
    def initialize_node_classes(
            self, N: int, f_m: float, rng: np.random.Generator):
        assert hasattr(self, 'set_node_class'),\
            "This Mixin requires a Graph object"
        self.set_node_class(
            CLASS_ATTRIBUTE,
            BinaryClassNodeVector.from_fraction(
                N=N,
                f_m=f_m,
                rng=rng))

    def get_minority_class(self) -> BinaryClassNodeVector:
        return self.get_node_class(CLASS_ATTRIBUTE)

class BinaryClassGraph(Graph, BinaryClassGraphMixin): pass
class BinaryClassDiGraph(DiGraph, BinaryClassGraphMixin): pass
