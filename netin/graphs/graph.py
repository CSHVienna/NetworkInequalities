from typing import Any, Optional, Dict, Callable, Hashable
from collections import defaultdict

import numpy as np
import networkx as nx

from .event import Event
from .node_vector import NodeVector
from .categorical_node_vector import CategoricalNodeVector
from ..base_class import BaseClass

class Graph(BaseClass):
    graph: nx.Graph
    _event_handlers: Dict[Event, Callable[[Any], None]]
    _node_attributes: Dict[str, CategoricalNodeVector]

    def __init__(self, *args, **attr) -> None:
        BaseClass.__init__(self)
        self._init_graph(*args, **attr)
        self._event_handlers = defaultdict(list)
        self._node_attributes = {}

    @classmethod
    def from_nxgraph(cls, graph: nx.Graph) -> "Graph":
        g = Graph()
        g.graph = graph
        return g

    def _init_graph(self, *args, **kwargs):
        self.graph = nx.Graph(*args, **kwargs)

    def set_node_attribute(self, name: str, node_vector: CategoricalNodeVector):
        assert len(node_vector) == len(self.graph),\
            f"Length of node vector `{name}` does not match the number of nodes in the graph (N={len(self.graph)})"
        self._node_attributes[name] = node_vector

    def get_node_attribute(self, name: str) -> CategoricalNodeVector:
        return self._node_attributes[name]

    def add_edge(self, source: Hashable, target: Hashable, **attr) -> None:
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
        for name, attr in self._node_attributes.items():
            d[self.__class__.__name__][name] = {}
            attr.get_metadata(d[self.__class__.__name__][name])
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
            Graph.assign_node_attributes(g_copy, node_attributes)
        return g_copy

    def assign_node_attributes(self, node_attributes: Dict[str, NodeVector]):

        Graph.assign_nx_node_attributes(self.graph, node_attributes)

    @staticmethod
    def assign_nx_node_attributes(
        graph: nx.Graph, node_attributes: Dict[str, NodeVector]):
        for name, node_vector in node_attributes.items():
            assert(len(node_vector) == len(graph)),\
                f"Length of node vector `{name}` does not match the number of nodes in the graph"
            nx.set_node_attributes(
                G=graph,
                name=name,
                values=node_vector.to_dict())

    @staticmethod
    def assign_nx_graph_class_attribute(
        graph: nx.Graph, node_classes: CategoricalNodeVector):
        assert(node_classes.name is not None),\
            "CategoricalNodeVector must have a name"
        graph.graph["class_attributes"] = node_classes.name
        graph.graph["class_labels"] = node_classes.class_labels
        graph.graph["class_values"] = list(range(node_classes.n_values))

    @staticmethod
    def get_node_attributes(
        graph: nx.Graph, name: str) -> NodeVector:
        nodes, values = zip(*nx.get_node_attributes(graph, name).items())
        return NodeVector.from_ndarray(
            N=len(graph),
            node_labels=nodes,
            values=np.asarray(values),
            name=name)

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
