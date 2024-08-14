############################################
# System dependencies
############################################
from typing import List
import networkx as nx
import numpy as np
import gc

############################################
# Local dependencies
############################################
from netin.models import DirectedModel
from netin.utils import validator as val
from netin.stats import networks as net
from netin.utils.constants import CLASS_ATTRIBUTE
from . import constants as const


############################################
# Class
############################################
class Sampling(object):
    """Base class for sampling methods.

    Parameters
    ----------
    g: netin.Graph | netin.DiGraph
        global network

    pseeds: float
        fraction of seeds to sample

    max_tries: int
        maximum number of tries to sample a subgraph with enough classes and edges

    random_seed: object
        seed for random number generator

    kwargs: dict
        additional parameters for the sampling method

    Notes
    -----
    - The original graph ``g`` (passed as parameter) is not modified.
       The sampling method creates a copy of it, and stores it in ``self.g``.
    - This class does not create a subgraph.

    """

    def __init__(self,
                 graph: nx.DiGraph,
                 pseeds: float,
                 class_attribute: str = CLASS_ATTRIBUTE,
                 max_tries: int = const.MAX_TRIES,
                 random_seed: object = None, **kwargs):
        self.g = graph
        self.class_attribute = class_attribute
        self.pseeds = pseeds
        self.max_tries = max_tries
        self.random_seed = random_seed
        self.nseeds = int(pseeds * self.g.number_of_nodes())
        self.sample = None
        self.nodes = None
        self.train_index = None
        self.test_index = None
        self.membership_y = None
        self.feature_x = None
        self.test_nodes = None
        self.kwargs = kwargs
        np.random.seed(self.random_seed)

    @classmethod
    def from_directed_model(
        cls,
        model: DirectedModel, pseeds: float,
        class_attribute: str = CLASS_ATTRIBUTE,
        max_tries: int = const.MAX_TRIES,
        random_seed: object = None, **kwargs) ->\
            "Sampling":
        graph = model.graph.to_nxgraph(node_attributes=model.node_attributes[class_attribute])
        graph.graph['model'] = model.__class__.__name__
        graph.graph["class_values"] = model.node_attributes[class_attribute].get_class_values()
        return cls(
            graph=graph,
            class_attribute=class_attribute,
            pseeds=pseeds,
            max_tries=max_tries,
            random_seed=random_seed,
            **kwargs)

    def sampling(self):
        """
        Creates a new instance of the respective sampling method, and calls its respective extract_subgraph method.
        """
        val.validate_float(self.pseeds, 0, 1)
        val.validate_more_than_one(net.get_node_attributes(self.g))
        self.sample = nx.DiGraph() if self.g.is_directed() else nx.Graph()
        self._extract_subgraph()
        self._set_graph_metadata()

    @property
    def method_name(self) -> str:
        """
        Name of sampling method.
        """
        return ''

    def _count_classes(self, nodes: List) -> int:
        """
        Counts the number of classes in a given set of nodes

        Parameters
        ----------
        nodes: list
            list of nodes

        Returns
        -------
        int
            number of classes
        """
        return len(set([self.g.nodes[n][self.class_attribute] for n in nodes]))

    def _extract_subgraph(self):
        """
        Creates a subgraph from G based on the sampling technique
        """
        max_tries = self.max_tries or const.MAX_TRIES
        num_edges = 0
        tries = -1

        while num_edges < const.MIN_EDGES:

            tries += 1
            if tries >= max_tries:
                raise RuntimeWarning(f"The sample has not enough edges ({num_edges}), and max_tries has exceeded. "
                                     "Try increasing the number of tries or increasing the number of seeds.")
                return

            nodes, edges = self._sample()

            ### 2. recreate induced subgraph
            sample = self.g.copy()

            # removing edges
            if edges:
                edges_to_remove = set(sample.edges()) - edges
                sample.remove_edges_from(edges_to_remove)

            # removing nodes
            if nodes:
                nodes_to_remove = [n for n in self.g.nodes if n not in nodes]
                sample.remove_nodes_from(nodes_to_remove)

            num_edges = sample.number_of_edges()

        if num_edges < const.MIN_EDGES:
            raise RuntimeWarning("The sample has no edges.")

        self.sample = sample.copy()
        gc.collect()

    def _set_graph_metadata(self):
        """
        Updates the training sample subgraph metadata
        """
        self.sample.graph['method'] = self.method_name
        self.sample.graph['pseeds'] = self.pseeds
        nx.set_node_attributes(G=self.g, name='seed', values={n: int(n in self.sample) for n in self.g.nodes})
        self.sample.graph['m'] = net.get_min_degree(self.sample)
        self.sample.graph['d'] = nx.density(self.sample)
        self.sample.graph['n'] = self.sample.number_of_nodes()
        self.sample.graph['f_m'] = net.get_minority_fraction(self.sample)
        self.sample.graph['similarity'] = net.get_similitude(self.sample)
        self.sample.graph['e'] = self.sample.number_of_edges()
        k, km, kM = net.get_average_degrees(self.sample)
        self.sample.graph['k'] = k
        self.sample.graph['km'] = km
        self.sample.graph['kM'] = kM
        self.sample.graph['random_seed'] = self.random_seed
        self.sample.graph['original_graph'] = self.sample.graph['model']
        del (self.sample.graph['model'])
        self.sample.model_name = f"{self.sample.model_name}\n{self.method_name}"

        # for LINK: working with matrices
        self.nodes = list(self.g.nodes)
        self.train_index = np.array([i for i, n in enumerate(self.nodes) if n in self.sample])
        self.test_nodes, self.test_index = zip(*[(n, i) for i, n in enumerate(self.nodes) if n not in self.sample])
        self.test_index = np.array(self.test_index)
        self.feature_x = nx.adjacency_matrix(self.g, self.nodes).toarray()
        self.membership_y = np.array([
            self.g.graph['class_values'].index(
                self.g.nodes[n][self.class_attribute])\
                    for n in self.nodes])

    def info(self):
        """
        Prints a summary of the training sample subgraph, including its attributes.

        """
        print(nx.info(self.sample))
        print(self.sample.graph)
