import networkx as nx

from .graph import Graph

class DiGraph(Graph, nx.DiGraph):
    """Directed graph."""
