from typing import Any, Optional, Dict, Callable
from collections import defaultdict

import networkx as nx

from .event import Event
from ..base_class import BaseClass

class Graph(BaseClass):
    """_summary_

    Parameters
    ----------
    nx : _type_
        _description_
    BaseGraph : _type_
        _description_
    """
    graph: nx.Graph
    _event_handlers: Dict[Event, Callable[[Any], None]]

    def __init__(self, *args, **attr) -> None:
        BaseClass.__init__(self)
        self._init_graph(*args, **attr)
        self._event_handlers = defaultdict(list)

    def _init_graph(self, *args, **kwargs):
        self.graph = nx.Graph()

    def add_edge(self, source: int, target: int, **attr) -> None:
        self.trigger_event(source, target, event=Event.LINK_ADD_BEFORE)
        self.graph.add_edge(
            u_of_edge=source, v_of_edge=target, **attr)
        self.trigger_event(source, target, event=Event.LINK_ADD_AFTER)

    def register_event_handler(
        self, event: Event, function: Callable[[Any], None]):
        self._event_handlers[event].append(function)

    def trigger_event(self, *args, event: Event, **kwargs):
        for function in self._event_handlers[event]:
            function(*args, **kwargs)

    def get_metadata(self, d_meta_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        d = super().get_metadata(d_meta_data)
        d[self.__class__.__name__] = {
            "n_nodes": len(self),
            "n_edges": self.graph.number_of_edges(),
            "event_handlers" : {
                event: [f.__name__ for f in functions]\
                    for event, functions in self._event_handlers.items()}
        }
        return d

    def copy(self, as_view=False):
        g = self.graph.copy(as_view)
        g_copy = Graph()
        g_copy.graph = g
        for event, functions in self._event_handlers.items():
            g_copy.register_event_handler(event, functions)
        return g_copy
