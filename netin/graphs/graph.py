from typing import Any, Optional, Dict, Callable
from collections import defaultdict

import networkx as nx
import numpy as np

from .event import Event
from .node_vector import NodeVector
from ..base_class import BaseClass

class Graph(BaseClass):
    graph: nx.Graph
    _event_handlers: Dict[Event, Callable[[Any], None]]

    def __init__(self, *args, **attr) -> None:
        BaseClass.__init__(self)
        self._init_graph(*args, **attr)
        self._event_handlers = defaultdict(list)

    def _init_graph(self, *args, **kwargs):
        self.graph = nx.Graph(*args, **kwargs)

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

    def to_nxgraph(
            self,
            node_attributes: Optional[Dict[str, NodeVector]] = None) -> nx.Graph:
        g_copy = self.graph.copy()
        if node_attributes is not None:
            for name, node_vector in node_attributes.items():
                nx.set_node_attributes(
                    G=g_copy,
                    name=name,
                    # Create a dictionary with the node indices as keys
                    # and the values of the node vector as values
                    values=dict(zip(
                        np.arange(len(node_vector)), node_vector.vals())))
        return g_copy

    ################################################
    # Method forwards to networkx.Graph
    ################################################
    def add_node(self, node: int, **attr):
        return self.graph.add_node(node, **attr)

    def add_nodes_from(self, *args, **kwargs):
        return self.graph.add_nodes_from(*args, **kwargs)

    def is_directed(self):
        return self.graph.is_directed()

    def has_edge(self, source: int, target: int):
        return self.graph.has_edge(source, target)

    def number_of_nodes(self):
        return self.graph.number_of_nodes()

    def number_of_edges(self):
        return self.graph.number_of_edges()

    def degree(self):
        return self.graph.degree()

    def neighbors(self, node: int):
        return self.graph.neighbors(node)

    def __len__(self):
        return len(self.graph)

    def __getitem__(self, name: str) -> Any:
        return self.graph.__getitem__(name)
