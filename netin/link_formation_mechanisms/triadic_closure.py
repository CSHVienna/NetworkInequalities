import numpy as np

from netin.graphs import Graph
from netin.graphs.event import Event
from .link_formation_mechanism import LinkFormationMechanism

class TriadicClosure(LinkFormationMechanism):
    _a_friend_of_friends: np.ndarray
    _source_curr: int

    def __init__(self, graph: Graph) -> None:
        super().__init__(graph)
        self._source_curr = -1
        self.graph.register_event_handler(
            event=Event.LINK_ADD_AFTER,
            function=self._update_friends_of_friends)

    def _init_friends_of_friends(self, source: int):
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
        if source != self._source_curr:
            return
        for fof in self.graph.neighbors[target]:
            if fof not in self.graph.neighbors[source]:
                self._a_friend_of_friends[fof] = 1.

    def get_target_probabilities(self, source) -> np.ndarray:
        if source != self._source_curr:
            self._init_friends_of_friends(source=source)
        return self._a_friend_of_friends
