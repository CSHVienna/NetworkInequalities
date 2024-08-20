from typing import List, Tuple

import networkx as nx

from .graph import Graph
from .node_vector import NodeVector

class DiGraph(Graph):
    def is_directed(self) -> bool:
        return True

    @classmethod
    def from_nxgraph(cls, graph: nx.DiGraph,
            node_attributes_names: List[str] = None,
            sort_node_labels: bool = True) -> Tuple[NodeVector, "Graph"]:
        return super().from_nxgraph(
            graph=graph,
            node_attributes_names=node_attributes_names,
            sort_node_labels=sort_node_labels)

    def number_of_edges(self):
        # Graph stores edges as undirected, so we need to multiply by 2
        return super().number_of_edges() * 2

    def _add_edge(self, source: int, target: int):
        assert source in self._graph, f"Node {source} does not exist"
        assert target in self._graph, f"Node {target} does not exist"
        assert target not in self._graph[source],\
            f"Edge `({source},{target})` already exists"
        self._graph[source].add(target)
