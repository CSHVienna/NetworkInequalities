import networkx as nx

class DiGraph(nx.DiGraph):

    def __init__(self, incoming_graph_data=None, **attr):
        super(DiGraph, self).__init__(incoming_graph_data, **attr)
