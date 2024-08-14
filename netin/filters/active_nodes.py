import numpy as np

from .filter import Filter
from ..graphs.node_vector import NodeVector
from ..graphs.directed import DiGraph
from ..graphs.event import Event

class ActiveNodes(Filter):
    _out_degrees: NodeVector

    def __init__(self, N: int, graph: DiGraph) -> None:
        assert isinstance(graph, DiGraph),\
            f"`graph` should be directe but is of type `{type(graph)}`"
        super().__init__()
        self._out_degrees = NodeVector(N, dtype=int)
        graph.register_event_handler(
            event=Event.LINK_ADD_AFTER,
            function=self._update_out_degrees
        )

    def _update_out_degrees(self, source: int, _: int):
        self._out_degrees[source] += 1

    def get_target_mask(self, _: int) -> NodeVector:
        target_mask = np.where(
            self._out_degrees.attr() > 0, 1., 0.)
        return NodeVector.from_ndarray(target_mask)
