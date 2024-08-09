import numpy as np

from ..graphs.graph import Graph
from ..graphs.event import Event
from ..graphs.node_attributes import NodeAttributes
from .filter import Filter

class TriadicClosure(Filter):
    """
    A filter based on triadic closure.

    This filter assigns a zero probability to nodes who are not friend of friends of the source node.

    Attributes:
        _a_friend_of_friends (np.ndarray): Array representing the presence of friends of friends.
        _source_curr (int): Current source node.

    Methods:
        __init__(self, graph: Graph): Initializes the TriadicClosure object.
        _init_friends_of_friends(self, source: int): Initializes the array of friends of friends.
        _update_friends_of_friends(self, source: int, target: int): Updates the array of friends of friends.
        get_target_probabilities(self, source): Returns the probabilities of forming links to target nodes.
    """

    _a_friend_of_friends: np.ndarray
    _source_curr: int

    def __init__(self, graph: Graph) -> None:
        super().__init__()
        self._source_curr = -1
        self.graph = graph
        self.graph.register_event_handler(
            event=Event.LINK_ADD_AFTER,
            function=self._update_friends_of_friends)

    def _init_friends_of_friends(self, source: int):
        """
        Initializes the array of friends of friends.

        Args:
            source (int): The source node.

        Returns:
            None
        """
        self._source_curr = source

        self._a_friend_of_friends = np.zeros(len(self.graph))
        fof = [
            f_o_f\
                for friend in self.graph.neighbors(source)\
                    for f_o_f in self.graph.neighbors(friend)\
                        if f_o_f not in self.graph.neighbors(source)
        ]
        self._a_friend_of_friends[fof] = 1.

    def _update_friends_of_friends(self, source: int, target: int):
        """
        Updates the array of friends of friends after a link was formed between `source` and `target`.

        Args:
            source (int): The source node.
            target (int): The target node.

        Returns:
            None
        """
        if source != self._source_curr:
            return
        for fof in self.graph.neighbors[target]:
            if fof not in self.graph.neighbors[source]:
                self._a_friend_of_friends[fof] = 1.

    def get_target_mask(self, source) -> NodeAttributes:
        """
        Returns the probabilities of forming links to target nodes.

        Target probabilities are one for nodes that are friends of friends of the source node and zero otherwise.

        Args:
            source: The source node.

        Returns:
            np.ndarray: Array representing the probabilities of forming links to target nodes.
        """
        if source != self._source_curr:
            self._init_friends_of_friends(source=source)
        return NodeAttributes.from_ndarray(
            self._a_friend_of_friends)
