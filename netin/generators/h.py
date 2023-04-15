from collections import Counter
from typing import Union, Set

import numpy as np

from netin.utils import constants as const
from netin.utils import validator as val
from .graph import Graph


class Homophily(Graph):

    ############################################################
    # Constructor
    ############################################################

    def __init__(self, n: int, f_m: float, h_MM: float, h_mm: float, seed: object = None):
        """

        Parameters
        ----------
        n: int
            number of nodes (minimum=2)

        f_m: float
            fraction of minorities (minimum=1/n, maximum=(n-1)/n)

        h_MM: float
            homophily (similarity) between majority nodes (minimum=0, maximum=1.)

        h_mm: float
            homophily (similarity) between minority nodes (minimum=0, maximum=1.)

        attr: dict
            attributes to add to graph as key=value pairs

        Notes
        -----
        The initialization is a graph with n nodes and no edges.
        Then, everytime a node is selected as source, it gets connected to k target nodes.
        Target nodes are selected via preferential attachment (in-degree), and homophily (h_**)

        References
        ----------
        - [1] A. L. Barabasi and R. Albert "Emergence of scaling in random networks", Science 286, pp 509-512, 1999.
        """
        Graph.__init__(self, n=n, f_m=f_m, seed=seed)
        self.h_MM = h_MM
        self.h_mm = h_mm
        self.mixing_matrix = None

    ############################################################
    # Init
    ############################################################

    def _infer_model_name(self):
        """
        Infers the name of the model.
        """
        return self.set_model_name(const.H_MODEL_NAME)

    def _validate_parameters(self):
        """
        Validates the parameters of the undirected.
        """
        Graph._validate_parameters(self)
        val.validate_float(self.h_MM, minimum=0., maximum=1.)
        val.validate_float(self.h_mm, minimum=0., maximum=1.)

    def get_metadata_as_dict(self) -> dict:
        obj = Graph.get_metadata_as_dict(self)
        obj.update({
            'h_MM': self.h_MM,
            'h_mm': self.h_mm,
        })
        return obj

    ############################################################
    # Getters & Setters
    ############################################################

    def set_homophily_majority(self, h_MM):
        """
        Parameters
        ----------
        h_MM: float
            homophily (similarity) between majority nodes (minimum=0, maximum=1.)
        """
        self.h_MM = h_MM

    def get_homophily_majority(self):
        """
        Returns
        -------
        h_MM: float
            homophily (similarity) between majority nodes (minimum=0, maximum=1.)
        """
        return self.h_MM

    def set_homophily_minority(self, h_mm):
        """
        Parameters
        ----------
        h_mm: float
            homophily (similarity) between minority nodes (minimum=0, maximum=1.)
        """
        self.h_mm = h_mm

    def get_homophily_minority(self):
        """
        Returns
        -------
        h_mm: float
            homophily (similarity) between minority nodes (minimum=0, maximum=1.)
        """
        return self.h_mm

    def get_homophily_between_source_and_target(self, source, target):
        return self.mixing_matrix[self.node_labels[source], self.node_labels[target]]

    ############################################################
    # Generation
    ############################################################

    def _initialize(self, class_attribute: str = 'm', class_values: list = None, class_labels: list = None):
        Graph._initialize(self, class_attribute, class_values, class_labels)
        self.h_MM = val.calibrate_null_probabilities(self.h_MM)
        self.h_mm = val.calibrate_null_probabilities(self.h_mm)
        self.mixing_matrix = np.array([[self.h_MM, 1 - self.h_MM], [1 - self.h_mm, self.h_mm]])

    def get_target_probabilities(self, source: Union[None, int], target_set: Union[None, Set[int], np.array],
                                 special_targets: Union[None, object, iter] = None) -> tuple[np.array, set[int]]:
        probs = np.array([self.get_homophily_between_source_and_target(source, target) for target in target_set])
        probs /= probs.sum()
        return probs, target_set

    def get_target(self, source: Union[None, int], targets: Union[None, Set[int]],
                   special_targets: Union[None, object, iter]) -> int:
        """
        Picks a random target node based on the homophily/preferential attachment dynamic.

        Parameters
        ----------
        source: int
            Newly added node

        targets: Set[int]
            Potential target nodes in the graph

        Returns
        -------
            int: Target node that an edge should be added to
        """
        # Collect probabilities to connect to each node in target_list
        target_set = self.get_potential_nodes_to_connect(source, targets)
        probs = self.get_target_probabilities(source, target_set, special_targets)
        return np.random.choice(a=target_set, size=1, replace=False, p=probs)[0]

    ############################################################
    # Calculations
    ############################################################

    def info_params(self):
        print('h_MM: {}'.format(self.h_MM))
        print('h_mm: {}'.format(self.h_mm))
        print('mixing matrix: \n{}'.format(self.mixing_matrix))

    def info_computed(self):
        inferred_h_MM, inferred_h_mm = self.infer_homophily_values()
        print("- Empirical homophily within majority: {}".format(inferred_h_MM))
        print("- Empirical homophily within minority: {}".format(inferred_h_mm))

    def infer_homophily_values(self) -> (float, float):
        print('H', self.is_directed())
        e = self.count_edges_types()

        if self.is_directed():
            h_MM = e['MM'] / (e['MM'] + e['Mm'])
            h_mm = e['mm'] / (e['mm'] + e['mM'])
        else:
            h_MM = e['MM'] / (e['MM'] + e['Mm'] + e['mM'])
            h_mm = e['mm'] / (e['mm'] + e['mM'] + e['Mm'])

        return h_MM, h_mm
