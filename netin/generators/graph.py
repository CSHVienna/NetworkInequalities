from collections import Counter
from typing import Callable, Union, Set

import networkx as nx
import numpy as np

from netin.utils import constants as const
from netin.utils import validator as val


class Graph(nx.Graph):

    ############################################################
    # Constructor
    ############################################################

    def __init__(self, n: int, k: int, f_m: float, seed: object = None, **attr: object):
        """

        Parameters
        ----------
        n: int
            number of nodes (minimum=2)

        k: int
            minimum degree of nodes (minimum=1)

        f_m: float
            fraction of minorities (minimum=1/n, maximum=(n-1)/n)

        attr: dict
            attributes to add to graph as key=value pairs

        Notes
        -----
        The initialization is a graph with n nodes and no edges.
        Then, everytime a node is selected as source, it gets connected to k target nodes.
        Target nodes are selected via preferential attachment (in-degree), homophily (h_**),
        and/or triadic closure (tc).

        References
        ----------
        - [1] A. L. Barabasi and R. Albert "Emergence of scaling in random networks", Science 286, pp 509-512, 1999.
        """
        super().__init__(**attr)
        self.n = n
        self.k = k
        self.f_m = f_m
        self.seed = seed
        self.n_m = 0
        self.n_M = 0
        self.model_name = None
        self.class_attribute = None
        self.class_values = None
        self.class_labels = None
        self.node_list = None
        self.labels = None

    ############################################################
    # Init
    ############################################################

    def _infer_model_name(self):
        pass

    def _validate_parameters(self):
        """
        Validates the parameters of the graph.
        """
        val.validate_int(self.n, minimum=2)
        val.validate_int(self.k, minimum=1)
        val.validate_float(self.f_m, minimum=1 / self.n, maximum=(self.n - 1) / self.n)
        self.seed = self.seed if self.seed is not None else np.random.randint(0, 2 ** 32)

    def _set_class_info(self, class_attribute: str = 'm', class_values=None, class_labels=None):
        if class_labels is None:
            class_labels = [const.MAJORITY_LABEL, const.MINORITY_LABEL]
        if class_values is None:
            class_values = [0, 1]
        self.set_class_attribute(class_attribute)
        self.set_class_values(class_values)
        self.set_class_labels(class_labels)

    def get_metadata_as_dict(self) -> dict:
        """
        Returns metadata for a graph.
        """
        obj = {'name': self.get_model_name(),
               'class_attribute': self.get_class_attribute(),
               'class_values': self.get_class_values(),
               'class_labels': self.get_class_labels(),
               'n': self.n,
               'k': self.k,
               'f_m': self.f_m,
               'seed': self.seed}
        return obj

    ############################################################
    # Getters & Setters
    ############################################################

    def set_model_name(self, model_name):
        self.model_name = model_name

    def get_model_name(self):
        return self.model_name

    def set_class_attribute(self, class_attribute):
        self.class_attribute = class_attribute

    def get_class_attribute(self):
        return self.class_attribute

    def set_class_values(self, class_values):
        self.class_values = class_values

    def get_class_values(self):
        return self.class_values

    def set_class_labels(self, class_labels):
        self.class_labels = class_labels

    def get_class_labels(self):
        return self.class_labels

    ############################################################
    # Generation
    ############################################################

    def _initialize(self, class_attribute: str = 'm', class_values: list = None, class_labels: list = None):
        """
        Initializes the random seed and the graph metadata.
        """
        np.random.seed(self.seed)
        self._validate_parameters()
        self._init_graph(class_attribute, class_values, class_labels)
        self._init_nodes()

    def _init_graph(self, class_attribute: str = 'm', class_values: list = None, class_labels: list = None):
        """
        Sets the name of the model, class information, and the graph metadata.

        Parameters
        ----------
        class_attribute: str
            name of the class attribute

        class_values: list
            list of class values

        class_labels: list
            list of class labels
        """
        self._infer_model_name()
        self._set_class_info(class_attribute, class_values, class_labels)
        self.graph = self.get_metadata_as_dict()

    def _init_nodes(self):
        """
        Initializes the list of nodes with their respective labels.
        """
        self.node_list = np.arange(self.n)
        self.n_M = int(round(self.n * (1 - self.f_m)))
        self.n_m = self.n - self.n_M
        minorities = np.random.choice(self.node_list, self.n_m, replace=False)
        self.labels = {n: int(n in minorities) for n in self.node_list}

    def get_target_by_preferential_attachment(self, source: int, targets: Set[int], special_targets=None) -> int:
        """
        Picks a random target node based on preferential attachment.

        Parameters
        ----------
        source: int
            Newly added node

        targets: Set[int]
            Potential target nodes in the graph based on preferential attachment

        Returns
        -------
            int: Target node that an edge should be added to
        """
        # Collect probabilities to connect to each node in target_list
        target_list = [t for t in targets if t != source and t not in nx.neighbors(self, source)]
        probs = np.array([(self.degree(target) + const.EPSILON) for target in target_list])
        probs /= probs.sum()
        return np.random.choice(a=target_list, size=1, replace=False, p=probs)[0]

    def get_special_targets(self, source: int) -> object:
        pass

    def get_target(self, source: Union[None, int], targets: Union[None, Set[int]],
                   special_targets: Union[None, object, iter]) -> int:
        pass

    def update_special_targets(self, idx_target: int, source: int, target: int, targets: Set[int],
                               special_targets: Union[None, object, iter]):
        pass

    def on_edge_added(self, source: int, target: int):
        pass

    def generate(self):
        """
        A graph of n nodes is grown by attaching new nodes each with k edges.
        Each edge is either drawn by preferential attachment, homophily, and/or triadic closure.

        For triadic closure, a candidate is chosen uniformly at random from all triad-closing edges (of the new node).
        Otherwise, or if there are no triads to close, edges are connected via preferential attachment and/or homophily.

        Homophily varies ranges from 0 (heterophilic) to 1 (homophilic), where 0.5 is neutral.
        Similarly, triadic closure varies from 0 (no triadic closure) to 1 (full triadic closure).

        . PA: A graph with h_mm = h_MM in [0.5, None] and tc = 0 is a BA preferential attachment model.
        . PAH: A graph with h_mm not in [0.5, None] and h_MM not in [0.5, None] and tc = 0 is a PA model with homophily.
        . PATC: A graph with h_mm = h_MM in [0.5, None] and tc > 0 is a PA model with triadic closure.
        . PATCH: A graph with h_mm not in [0.5, None] and h_MM not in [0.5, None] and tc > 0 is a PA model
                 with homophily and triadic closure.

        Parameters
        ----------
        _on_edge_added: Union[None, Callable[[nx.Graph, int, int], None] (default=None)
            callback function to be called when an edge is added
            The function is expected to take the graph and two ints (source and target node) as input.
        """

        # 1. Init graph and nodes (assign class labels)
        self._initialize()
        self.add_nodes_from(self.node_list)
        nx.set_node_attributes(self, self.labels, self.class_attribute)

        # 3. Iterate until n nodes are added (starts with k pre-existing, unconnected nodes)
        for source in self.node_list[self.k:]:
            targets = set(range(source))  # targets via preferential attachment
            special_targets = self.get_special_targets(source)

            for idx_target in range(self.k):
                # Choose next target
                target = self.get_target(source, targets, special_targets)

                special_targets = self.update_special_targets(idx_target, source, target, targets, special_targets)

                # Finally add edge to graph
                self.add_edge(source, target)

                # Call event handlers if present
                self.on_edge_added(source, target)


    ############################################################
    # Calculations
    ############################################################

    def info_params(self):
        pass

    def info_computed(self):
        pass

    def info(self,
             _info_params_fnc: Union[None, Callable] = None,
             _info_computed_fnc: Union[None, Callable] = None, **kwargs):
        print("=== Params ===")
        print('n: {}'.format(self.n))
        print('k: {}'.format(self.k))
        print('f_m: {}'.format(self.f_m))
        self.info_params()
        print('seed: {}'.format(self.seed))

        print("=== Inferred ===")
        print('Model: {}'.format(self.get_model_name()))
        print('Class attribute: {}'.format(self.get_class_attribute()))
        print('Class values: {}'.format(self.get_class_values()))
        print('Class labels: {}'.format(self.get_class_labels()))

        print("=== Computed ===")
        print(f'- number of nodes: {self.number_of_nodes()}')
        print(f'- number of edges: {self.number_of_edges()}')
        print(f'- minimum degree: {self.calculate_minimum_degree()}')
        print(f'- fraction of minority: {self.calculate_fraction_of_minority()}')
        print(f'- edge-type counts: {self.count_edges_types()}')
        print(f"- density: {nx.density(self)}")
        print(f"- diameter: {nx.diameter(self)}")
        print(f"- average shortest path length: {nx.average_shortest_path_length(self)}")
        print(f"- average degree: {sum([d for n, d in self.degree]) / self.number_of_nodes()}")
        print(f"- degree assortativity: {nx.degree_assortativity_coefficient(self)}")
        print(f"- transitivity: {nx.transitivity(self)}")
        print(f"- average clustering: {nx.average_clustering(self)}")
        self.info_computed()

    def calculate_minimum_degree(self):
        return min([d for n, d in self.degree])

    def calculate_fraction_of_minority(self):
        return sum([1 for n, obj in self.nodes(data=True) if obj[self.class_attribute] == self.class_values[
            self.class_labels.index(const.MINORITY_LABEL)]]) / self.number_of_nodes()

    def count_edges_types(self):
        return Counter([
                           f"{self.class_labels[self.nodes[e[0]][self.class_attribute]]}{self.class_labels[self.nodes[e[1]][self.class_attribute]]}"
                           for e in self.edges])
