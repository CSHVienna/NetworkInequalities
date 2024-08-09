import numpy as np

from .filter import Filter
from ..graphs.node_attributes import NodeAttributes
from ..graphs.directed import DiGraph
from ..graphs.event import Event

class ActiveNodes(Filter):
    _out_degrees: NodeAttributes

    def __init__(self, N: int, graph: DiGraph) -> None:
        super().__init__()
        self._out_degrees = NodeAttributes(N, dtype=int)
        graph.register_event_handler(
            event=Event.LINK_ADD_AFTER,
            function=self._update_out_degrees
        )

    def _update_out_degrees(self, source: int, _: int):
        self._out_degrees[source] += 1

    def get_target_mask(self, _: int) -> NodeAttributes:
        target_mask = np.where(
            self._out_degrees.attr() > 0, 1., 0.)
        return NodeAttributes.from_ndarray(target_mask)
