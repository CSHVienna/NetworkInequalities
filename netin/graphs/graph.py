from typing import\
    Any, Optional, Dict,\
    Set, List, Tuple, Generator

import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix

from .node_vector import NodeVector
from .categorical_node_vector import CategoricalNodeVector
from ..utils.event_handling import HasEvents, Event
from ..base_class import BaseClass

class Graph(HasEvents, BaseClass):
    """A Graph representation that allows simple manipulation.
     It supports event handling and import/export functionality to other popular Graph libraries.
    """
    EVENTS = [Event.LINK_ADD_BEFORE, Event.LINK_ADD_AFTER]
    _graph: Dict[int, Set[int]]
    _node_classes: Dict[str, CategoricalNodeVector]

    def __init__(self, *args, **attr):
        BaseClass.__init__(self, **attr)
        HasEvents.__init__(self)
        self._init_graph(*args, **attr)
        self._node_classes = {}

    @classmethod
    def from_nxgraph(
            cls, graph: nx.Graph,
            node_attributes_names: List[str] = None,
            sort_node_labels: bool = True) -> Tuple[NodeVector, "Graph"]:
        """Take a `networkx.Graph` and return a `Graph` object and mapping of the nodes.
        NetworkX graphs uses custom node labels while `Graph` identifies nodes
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
                g.set_node_class(
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

    def set_node_class(
            self, name: str, node_vector: CategoricalNodeVector):
        """Set a node class.
        If the class already exists, it will be overwritten.

        Parameters
        ----------
        name : str
            Key under which the node class will be stored.
        node_vector : CategoricalNodeVector
            The node class.
        """
        self._node_classes[name] = node_vector

    def get_node_class(self, name: str) -> CategoricalNodeVector:
        """Get a node class by name.

        Parameters
        ----------
        name : str
            The name of the node class.

        Returns
        -------
        CategoricalNodeVector
            The node class.
        """
        return self._node_classes[name]

    def get_node_classes(self) -> Dict[str, CategoricalNodeVector]:
        """Get all node classes.

        Returns
        -------
        Dict[str, CategoricalNodeVector]
            The original dictionary containing the node classes.
        """
        return self._node_classes

    def has_node_class(self, name: str) -> bool:
        """Check if a node class exists.
        Identical to `name in self.get_node_classes()`.

        Parameters
        ----------
        name : str
            The name of the node class.

        Returns
        -------
        bool
            True if the node class exists, False otherwise.
        """
        return name in self._node_classes

    def add_edge(self, source: int, target: int) -> None:
        """Add an edge to the graph..
        This will trigger the `Event.LINK_ADD_BEFORE` and `Event.LINK_ADD_AFTER` events.

        Parameters
        ----------
        source : int
            The source node.
        target : int
            The target node.

        Events
        ------
        :attr:`.Event.LINK_ADD_BEFORE`
            Before a link is added to the graph.
        :attr:`.Event.LINK_ADD_AFTER`
            After a link is added to the graph.
        """
        self.trigger_event(source, target, event=Event.LINK_ADD_BEFORE)
        self._add_edge(source, target)
        self.trigger_event(source, target, event=Event.LINK_ADD_AFTER)

    def get_metadata(self, d_meta_data: Optional[Dict[str, Any]] = None)\
        -> Dict[str, Any]:
        """Returns the metadata of the Graph.

        Returns
        -------
        Dict[str, Any]
            Description of the metadata, including the number of nodes and edges.
        """
        d = super().get_metadata(d_meta_data)
        d[self.__class__.__name__] = {
            "n_nodes": len(self),
            "n_edges": self.number_of_edges(),
        }
        for name, attr in self._node_classes.items():
            d[self.__class__.__name__][name] = {}
            attr.get_metadata(d[self.__class__.__name__][name])
        return d

    def copy(self) -> "Graph":
        """Returns a copy of the Graph.
        This will copy the graph structure, node classes and event handlers.
        """
        g_copy = Graph()
        for node in range(len(self)):
            g_copy.add_node(node)
        for source, targets in self._graph.items():
            for target in targets:
                g_copy.add_edge(source, target)
        for event, functions in self._event_handlers.items():
            g_copy.register_event_handler(event, functions)
        for name, attr in self._node_classes.items():
            g_copy.set_node_class(name, attr.copy())
        return g_copy

    def to_nxgraph(self) -> nx.Graph:
        """Convert the Graph to a `networkx.Graph`.

        Returns
        -------
        nx.Graph
            The `networkx.Graph` representation of the graph.
        """
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
        """Add a node to the graph.

        Parameters
        ----------
        node : int
            The node to be added.
        """
        assert node not in self._graph, f"Node {node} already exists"
        self._graph[node] = set()

    def is_directed(self) -> bool:
        """Returns whether the graph is directed (always `False`).

        Returns
        -------
        bool
            Returns False.
        """
        return False

    def has_edge(self, source: int, target: int) -> bool:
        """Check if an edge exists between two nodes.

        Parameters
        ----------
        source : int
            The source node.
        target : int
            The target node.

        Returns
        -------
        bool
            True if the edge exists, False otherwise.
        """
        assert target in self._graph, f"Node {target} does not exist"
        return target in self._graph[source]

    def number_of_nodes(self) -> int:
        """Returns the number of nodes in the graph.

        Returns
        -------
        int
            The number of nodes in the graph.
        """
        return len(self)

    def number_of_edges(self) -> int:
        """Returns the number of edges in the graph.

        Returns
        -------
        int
            The number of edges in the graph.
        """
        return sum(len(targets) for targets in self._graph.values()) // 2

    def degree(self, node: int) -> int:
        """Returns the degree of a node.
        The degree is defined by the number of neighbors of a node.

        Parameters
        ----------
        node : int
            The node to calculate the degree for.

        Returns
        -------
        int
            The degree of the node.
        """
        return len(self.neighbors(node))

    def degrees(self) -> NodeVector:
        """Returns the degrees of all nodes.

        Returns
        -------
        NodeVector
            A NodeVector containing the degrees of all nodes.
        """
        return NodeVector.from_ndarray(
            np.array([self.degree(node) for node in range(len(self))]),
            name="degrees")

    def nodes(self) -> List[int]:
        """Returns a list of all nodes.

        Returns
        -------
        List[int]
            A list of all nodes.
        """
        return list(self._graph.keys())

    def edges(self) -> Generator[Tuple[int, int], None, None]:
        """Returns a generator of all edges.

        Yields
        ------
        Generator[Tuple[int, int], None, None]
            A generator of all edges. Yields tuples of the form `(source, target)`.
        """
        for source, targets in self._graph.items():
            for target in targets:
                if target > source:
                    continue
                yield source, target

    def neighbors(self, node: int) -> Set[int]:
        """Returns the neighbors of a node.

        Parameters
        ----------
        node : int
            The node to get the neighbors for.

        Returns
        -------
        Set[int]
            A set of neighbors.
        """
        return self._graph[node]

    def get_adjacency_matrix(self) -> csr_matrix:
        rows = []
        cols = []
        data = []

        for node, neighbors in self._graph.items():
            for neighbor in neighbors:
                rows.append(node)
                cols.append(neighbor)
                data.append(1)  # Assuming an unweighted adjacency matrix, use 1 for each edge

        # Create a sparse CSR matrix
        n_nodes = max(max(self._graph.keys()),
                      max(max(neighbors) for neighbors in self._graph.values())) + 1  # Get the size of the matrix
        adj_matrix = csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
        return adj_matrix

    def __len__(self):
        return len(self._graph)

    def __getitem__(self, node: int) -> Any:
        return self._graph.__getitem__(node)
