import networkx as nx

from .graph import Graph

class DiGraph(Graph):
    def _init_graph(self, *args, **kwargs):
        self.graph = nx.DiGraph(*args, **kwargs)

    ################################################
    # Method forwards to networkx.DiGraph
    ################################################
    def in_degree(self):
        return self.graph.in_degree()