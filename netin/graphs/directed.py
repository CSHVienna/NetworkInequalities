import networkx as nx

from .graph import Graph

class DiGraph(nx.DiGraph, Graph):
    """Directed graph."""
    def __init__(self, incoming_graph_data=None, **attr):
        Graph.__init__(self)
        nx.DiGraph.__init__(self, incoming_graph_data, **attr)
