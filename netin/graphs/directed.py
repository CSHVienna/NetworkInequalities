from typing import List, Tuple

import networkx as nx

from .graph import Graph
from .node_vector import NodeVector

class DiGraph(Graph):
    """Represents a directed graph.
    """
    def is_directed(self) -> bool:
        """Returns True.

        Returns
        -------
        bool
            Always True.
        """
        return True

    @classmethod
    def from_nxgraph(cls, graph: nx.DiGraph,
            node_attributes_names: List[str] = None,
            sort_node_labels: bool = True) -> Tuple[NodeVector, "Graph"]:
        """Creates a DiGraph from a `nx.DiGraph`.

        Returns
        -------
        Tuple[NodeVector, DiGraph]
            A NodeVector with the original node labels and the DiGraph.
            Because NetworkX supports custom indexing, the node labels
            may not be the same as the original ones.
            For this reason, the original node labels are returned, mapped
            to the new ones.
            The original label of node `i` can be accessed by `node_labels[i]`.
        """
        return super().from_nxgraph(
            graph=graph,
            node_attributes_names=node_attributes_names,
            sort_node_labels=sort_node_labels)

    def number_of_edges(self) -> int:
        """Returns the number of edges in the graph.

        Returns
        -------
        int
            The number of edges in the graph.
        """
        # Graph stores edges as undirected, so we need to multiply by 2
        return super().number_of_edges() * 2

    def _add_edge(self, source: int, target: int):
        """Adds a directed edge to the graph.

        Parameters
        ----------
        source : int
            The source node
        target : int
            The target node
        """
        assert source in self._graph, f"Node {source} does not exist"
        assert target in self._graph, f"Node {target} does not exist"
        assert target not in self._graph[source],\
            f"Edge `({source},{target})` already exists"
        self._graph[source].add(target)
