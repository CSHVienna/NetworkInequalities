from typing import Iterable, Any, Optional, Dict, Callable
from collections import defaultdict

import networkx as nx

from .event import Event
from ..base_class import BaseClass

class Graph(BaseClass, nx.Graph):
    """_summary_

    Parameters
    ----------
    nx : _type_
        _description_
    BaseGraph : _type_
        _description_
    """
    _event_handlers: Dict[Event, Callable[[Any], None]]

    def __init__(self, **attr) -> None:
        nx.Graph.__init__(
            self, incoming_graph_data=None, **attr)
        BaseClass.__init__(self)

        self._event_handlers = defaultdict(list)

    def add_edge(self, source: int, target: int, **attr) -> None:
        self.trigger_event(source, target, event=Event.LINK_ADD_BEFORE)
        nx.Graph.add_edge(self, source, target, **attr)
        self.trigger_event(source, target, event=Event.LINK_ADD_AFTER)

    def add_edges_from(self, ebunch_to_add: Any, **attr) -> None:
        for edge in ebunch_to_add:
            if len(edge) == 3:
                u,v,d = edge
                # Precedence of **d over **attr is specified in super-method
                self.add_edge(u,v, **d)
            elif len(edge) == 2:
                u,v, = edge
                self.add_edge(u,v, **attr)
            else:
                raise RuntimeError(f"Edges should be tuple or triplet, but got `{edge}`")

    def add_weighted_edges_from(
            self,
            ebunch_to_add: Iterable[tuple[int, int, float]],
            weight: str ='weight', **attr) -> None:
        self.add_edges_from(ebunch_to_add=[(u,v,{weight: w, **attr}) for u,v,w in ebunch_to_add])

    def register_event_handler(
        self, event: Event, function: Callable[[Any], None]):
        self._event_handlers[event].append(function)

    def trigger_event(self, *args, event: Event, **kwargs):
        for function in self._event_handlers[event]:
            function(*args, **kwargs)

    def get_metadata(self, d_meta_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        d = super().get_metadata(d_meta_data)
        d[self.__class__.__name__] = {
            "is_directed": self.is_directed(),
            "n_nodes": len(self),
            "n_edges": self.number_of_edges(),
            "event_handlers" : {
                event: [f.__name__ for f in functions]\
                    for event, functions in self._event_handlers.items()}
        }
        return d

    def copy(self, as_view=False):
        g = super().copy(as_view)
        for event, functions in self._event_handlers.items():
            g.register_event_handler(event, functions)
        return g
