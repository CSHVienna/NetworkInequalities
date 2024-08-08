import networkx as nx

from .graph import Graph

class DiGraph(Graph):
    def _init_graph(self, *args, **kwargs):
        self.graph = nx.DiGraph(*args, **kwargs)
