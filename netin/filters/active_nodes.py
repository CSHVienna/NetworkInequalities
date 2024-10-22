import numpy as np

from .filter import Filter
from ..graphs.node_vector import NodeVector
from ..graphs.directed import DiGraph
from ..utils.event_handling import Event

class ActiveNodes(Filter):
    """A filter for active nodes.
    A node is considered active if it has at least one outgoing link.

    Parameters
    ----------
    N : int
        Number of nodes.
    graph : DiGraph
        The graph used to update node activity.
    """
    _nodes_active: NodeVector

    def __init__(self, N: int, graph: DiGraph) -> None:
        assert isinstance(graph, DiGraph),\
            f"`graph` should be directe but is of type `{type(graph)}`"
        super().__init__()
        self._nodes_active = NodeVector.from_ndarray(
            np.zeros(N, dtype=bool))
        graph.register_event_handler(
            event=Event.LINK_ADD_AFTER,
            function=self._update_out_degrees
        )

    def _update_out_degrees(self, source: int, _: int):
        self._nodes_active[source] = True

    def get_target_mask(self, _: int) -> NodeVector:
        """Returns a mask for the target nodes.

        Parameters
        ----------
        _ : int
            Unused.

        Returns
        -------
        NodeVector
            A filter mask for the target nodes.
        """
        target_filter = np.where(
            self._nodes_active, 1., 0.)
        return NodeVector.from_ndarray(target_filter)
