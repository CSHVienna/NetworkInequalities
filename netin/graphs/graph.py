from typing import\
    Any, Optional, Dict,\
    Set, List, Tuple
from collections import defaultdict

import numpy as np
import networkx as nx

from ..event import Event
from .node_vector import NodeVector
from .categorical_node_vector import CategoricalNodeVector
from ..base_class import BaseClass

class Graph(BaseClass):
    __events = [Event.LINK_ADD_BEFORE, Event.LINK_ADD_AFTER]
    _graph: Dict[int, Set[int]]
    _node_classes: Dict[str, CategoricalNodeVector]

    def __init__(self, *args, **attr) -> None:
        BaseClass.__init__(self)
        self._init_graph(*args, **attr)
        self._event_handlers = defaultdict(list)
        self._node_classes = {}

    @classmethod
    def from_nxgraph(
            cls, graph: nx.Graph,
            node_attributes_names: List[str] = None,
            sort_node_labels: bool = True) -> Tuple[NodeVector, "Graph"]:
        """Take a `networkx.Graph` and return a `Graph` object and mapping of the nodes.
        NetworkX graphs uses custom node labels while `Graph` uses identifies nodes
        by integer indices.
        This method translates the node labels to integer indices and returns the
        mapping as a `NodeVector` where the value `v_i` at index `i` is the node
        label of node `i`.

        Parameters
        ----------
        graph : nx.Graph
            The `networkx.Graph` object to be converted.
        node_attributes_names : List[str], optional
            List of node attributes to be included in the `Graph` object.
            Each element is the name of `node` attribute in the `networkx.Graph`,
            retrievable by `nx.get_node_attributes(graph, name)`.
            Values must be integers.
        sort_node_labels : bool, optional
            Whether to sort the node labels in ascending order. Default is `True`.
            If the node labels of the `nx_Graph` are a full integer range, the node
            mapping will be identical (the returned `NodeVector` will be identical
            to `np.arange(len(graph))`).

        Returns
        -------
        Tuple[NodeVector, Graph]
            A tuple containing the node labels and the Graph object.
        """
        g = Graph()
        nx_node_labels = sorted(list(graph.nodes)) if sort_node_labels else list(graph.nodes)
        for node in nx_node_labels:
            g.add_node(node)
        for source, targets in graph.edges:
            g.add_edge(source, targets)
        if node_attributes_names is not None:
            for name in node_attributes_names:
                nx_node_attr = nx.get_node_attributes(graph, name)
                g.set_node_attribute(
                    name=name,
                    node_vector=CategoricalNodeVector.from_ndarray(
                        values=np.asarray([nx_node_attr[node] for node in nx_node_labels]),
                        name=name))
        nv_node_labels = NodeVector.from_ndarray(
            values=np.asarray(nx_node_labels),
            name="node_labels")
        return nv_node_labels, g

    def _init_graph(self):
        self._graph = {}

    def _add_edge(self, source: int, target: int):
        assert source in self._graph, f"Node {source} does not exist"
        assert target in self._graph, f"Node {target} does not exist"
        assert target not in self._graph[source],\
            f"Edge `({source},{target})` already exists"
        self._graph[source].add(target)
        self._graph[target].add(source)

    def set_node_attribute(
            self, name: str, node_vector: CategoricalNodeVector):
        self._node_classes[name] = node_vector

    def get_node_attribute(self, name: str) -> CategoricalNodeVector:
        return self._node_classes[name]

    def add_edge(self, source: int, target: int) -> None:
        self.trigger_event(source, target, event=Event.LINK_ADD_BEFORE)
        self._add_edge(source, target)
        self.trigger_event(source, target, event=Event.LINK_ADD_AFTER)

    def get_metadata(self, d_meta_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        d = super().get_metadata(d_meta_data)
        d[self.__class__.__name__] = {
            "n_nodes": len(self),
            "n_edges": self.number_of_edges(),
        }
        for name, attr in self._node_classes.items():
            d[self.__class__.__name__][name] = {}
            attr.get_metadata(d[self.__class__.__name__][name])
        return d

    def copy(self):
        g_copy = Graph()
        for node in range(len(self)):
            g_copy.add_node(node)
        for source, targets in self._graph.items():
            for target in targets:
                g_copy.add_edge(source, target)
        for event, functions in self._event_handlers.items():
            g_copy.register_event_handler(event, functions)
        for name, attr in self._node_classes.items():
            g_copy.set_node_attribute(name, attr.copy())
        return g_copy

    def to_nxgraph(self) -> nx.Graph:
        g = nx.Graph() if not self.is_directed() else nx.DiGraph()
        for i in range(len(self)):
            g.add_node(i)
        for source, targets in self._graph.items():
            for target in targets:
                g.add_edge(source, target)

        for name, node_vector in self._node_classes.items():
            nx.set_node_attributes(G=g, name=name, values={
                i: node_vector[i] for i in range(len(self))})
        return g

    def add_node(self, node: int):
        assert node not in self._graph, f"Node {node} already exists"
        self._graph[node] = set()

    def is_directed(self) -> bool:
        return False

    def has_edge(self, source: int, target: int):
        assert target in self._graph, f"Node {target} does not exist"
        return target in self._graph[source]

    def number_of_edges(self):
        return sum(len(targets) for targets in self._graph.values()) // 2

    def __len__(self):
        return len(self._graph)

    def __getitem__(self, node: int) -> Any:
        return self._graph.__getitem__(node)
