import numpy as np

from ..graphs.graph import Graph
from ..utils.event_handling import Event
from ..graphs.node_vector import NodeVector
from .link_formation_mechanism import LinkFormationMechanism

class TriadicClosure(LinkFormationMechanism):
    """The Triadic Closure link formation mechanism.
    Friends of friends are assigned a uniform probability
    of forming a link, while other nodes have zero probability.

    Parameters
    ----------
    N : int
        Number of nodes.
    graph : Graph
        The graph used to keep track of friends of friends.
    """
    _a_friend_of_friends: np.ndarray
    _source_curr: int

    def __init__(self, N: int, graph: Graph) -> None:
        super().__init__(N=N)
        assert N >= len(graph),\
            "The number of nodes must be greater or equals to the number of nodes in the graph."
        self._source_curr = -1
        self.graph = graph
        self.graph.register_event_handler(
            event=Event.LINK_ADD_AFTER,
            function=self._update_friends_of_friends)

    def _init_friends_of_friends(self, source: int):
        """Initializes the array of friends of friends.

        Parameters
        ----------
        source : int
            The source node.
        """
        self._source_curr = source

        self._a_friend_of_friends = np.zeros(self.N)
        fof = [
            f_o_f\
                for friend in self.graph.neighbors(source)\
                    for f_o_f in self.graph.neighbors(friend)\
                        if (f_o_f not in self.graph.neighbors(source)) and (f_o_f != source)
        ]
        self._a_friend_of_friends[fof] = 1.

    def _update_friends_of_friends(self, source: int, target: int):
        """Updates the array of friends of friends after a link
        was formed between `source` and `target`.

        Parameters
        ----------
        source : int
            The source node.
        target : int
            The target node.
        """
        if source != self._source_curr:
            return
        for fof in self.graph.neighbors(target):
            if (fof not in self.graph.neighbors(source))\
                and (source != fof):
                self._a_friend_of_friends[fof] = 1.

    def _get_target_probabilities(self, source: int) -> NodeVector:
        """Returns the probabilities of forming links to target nodes.

        Target probabilities are uniform for nodes that are
        friends of friends of the source node and zero otherwise.

        Parameters
        ----------
        source : int
            The source node.

        Returns
        -------
        NodeVector
            Array representing the probabilities of forming links to target nodes.
        """
        if source != self._source_curr:
            self._init_friends_of_friends(source=source)
        if not np.any(self._a_friend_of_friends != 0.):
            return self._get_uniform_target_probabilities(source)
        return NodeVector.from_ndarray(
            self._a_friend_of_friends)
