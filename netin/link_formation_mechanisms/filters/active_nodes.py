import numpy as np

from ..link_formation_mechanism import LinkFormationMechanism
from ...graphs.node_attributes import NodeAttributes
from ...graphs.directed import DiGraph
from ...graphs.event import Event

class ActiveNodes(LinkFormationMechanism):
    _out_degrees: NodeAttributes

    def __init__(self, N: int, graph: DiGraph) -> None:
        super().__init__()
        self.out_degrees = NodeAttributes(N, dtype=int)
        graph.register_event_handler(
            event=Event.LINK_ADD_AFTER,
            function=self._update_out_degrees
        )

    def _update_out_degrees(self, source: int, _: int):
        self._out_degrees[source] += 1

    def get_target_probabilities(self, _: int) -> np.ndarray:
        target_probabilities = np.where(
            np.where(self._out_degrees.attr() > 0, 1., 0.)
        )
        return target_probabilities / target_probabilities.sum()
