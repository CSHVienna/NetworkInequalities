from .preferential_attachment import PreferentialAttachment
from ..graphs.graph import Graph
from ..utils.event_handling import Event
from ..graphs.node_vector import NodeVector

class InDegreePreferentialAttachment(PreferentialAttachment):
    """
    The preferential attachment link formation mechanism based on nodes' in-degree.
    """
    _a_degree: NodeVector

    def __init__(self,
                 graph: Graph,
                 n: int,
                 init_degrees: bool = True) -> None:
        assert graph.is_directed(), "The graph must be directed."
        super().__init__(N=len(graph), graph=graph)

        self._a_degree = NodeVector(
            n, dtype=int, name="in_degrees")
        if init_degrees:
            self.initialize_degree_array()

        self.graph.register_event_handler(
            event=Event.LINK_ADD_AFTER,
            function=self._update_degree_by_link)

    def initialize_degree_array(self):
        """Initializes the degree array.
        This is useful for when an existing graph is loaded and the degrees need to be computed.
        """
        for i,k in self.graph.in_degree():
            self._a_degree[i] = k

    def _update_degree_by_link(self, _: int, target: int):
        """
        Updates the degree of nodes when a new link is added.

        Args:
            source (int): The source node of the new link.
            target (int): The target node of the new link.

        Returns:
            None
        """
        self._a_degree[target] += 1
