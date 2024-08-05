import networkx as nx

from .graph import Graph

class DiGraph(nx.DiGraph, Graph):
    """Directed graph."""
