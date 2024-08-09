import numpy as np

from .filter import Filter
from ..graphs.graph import Graph
from ..graphs.node_attributes import NodeAttributes

class NoDoubleLinks(Filter):
    N: int
    graph: Graph

    def __init__(self, N: int, graph: Graph) -> None:
        super().__init__()
        self.N = N
        self.graph = graph

    def get_target_mask(self, source: int) -> NodeAttributes:
        target_mask = np.ones(self.N)
        target_mask[self.graph[source]] = 0.
        return NodeAttributes.from_ndarray(target_mask)
