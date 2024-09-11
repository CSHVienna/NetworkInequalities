from scipy.sparse import lil_matrix
from scipy.sparse import triu

from ...base_class import BaseClass
from ...graphs.graph import Graph
from ...utils import constants


class JanusGraph(BaseClass):

    def __init__(self, graph: Graph, **attr):
        BaseClass.__init__(self, **attr)
        self.graph = graph  # netin.Graph
        self.adj_matrix: lil_matrix = None
        self.adj_matrix: lil_matrix = None
        self.adj_matrix_clean: lil_matrix = None

    def init_data(self, is_global: bool = True):
        self.adj_matrix = self.graph.get_adjacency_matrix().tolil() * 100
        if is_global:
            if self.is_directed():
                # flatten 1 x n^2
                self.adj_matrix_clean = self.adj_matrix.reshape(1, -1)
            else:
                # just one side (eg. upper diagonal)
                self.adj_matrix_clean = triu(self.adj_matrix, k=1)
        else:
            self.adj_matrix_clean = self.adj_matrix

    def is_directed(self):
        return self.graph.is_directed()

    def number_of_nodes(self):
        return self.graph.number_of_nodes()

    def get_node_classes(self):
        return self.graph.get_node_classes()

    def get_node_class(self):
        return self.graph.get_node_class(constants.CLASS_ATTRIBUTE)
