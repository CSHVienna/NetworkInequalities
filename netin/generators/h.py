from typing import Union, Set, Tuple

import numpy as np

from netin.utils import constants as const
from netin.utils import validator as val
from .graph import Graph


class Homophily(Graph):
    """Class to model homophily as a mechanism of edge formation given a source and a target node.

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

    Notes
    -----
    This class does not generate a graph.
    """

    ############################################################
    # Constructor
    ############################################################

    def __init__(self, n: int, f_m: float, h_MM: float, h_mm: float, seed: object = None):
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
        Validates the parameters of the graph.
        """
        Graph._validate_parameters(self)
        val.validate_float(self.h_MM, minimum=0., maximum=1.)
        val.validate_float(self.h_mm, minimum=0., maximum=1.)

    def get_metadata_as_dict(self) -> dict:
        """
        Returns the metadata info (input parameters of the model) of the graph as a dictionary.

        Returns
        -------
        obj dict
            dictionary with the metadata info of the graph.
        """
        obj = Graph.get_metadata_as_dict(self)
        obj.update({
            'h_MM': self.h_MM,
            'h_mm': self.h_mm,
        })
        return obj

    ############################################################
    # Getters & Setters
    ############################################################

    def set_homophily_majority(self, h_MM: float):
        """
        Sets the homophily value between majority nodes.

        Parameters
        ----------
        h_MM: float
            homophily (similarity) between majority nodes (minimum=0, maximum=1.)
        """
        self.h_MM = h_MM

    def get_homophily_majority(self) -> float:
        """
        Returns the homophily value between majority nodes.

        Returns
        -------
        h_MM: float
            homophily (similarity) between majority nodes (minimum=0, maximum=1.)
        """
        return self.h_MM

    def set_homophily_minority(self, h_mm: float):
        """
        Sets the homophily value between minority nodes.

        Parameters
        ----------
        h_mm: float
            homophily (similarity) between minority nodes (minimum=0, maximum=1.)
        """
        self.h_mm = h_mm

    def get_homophily_minority(self) -> float:
        """
        Returns the homophily value between minority nodes.

        Returns
        -------
        h_mm: float
            homophily (similarity) between minority nodes (minimum=0, maximum=1.)
        """
        return self.h_mm

    def get_homophily_between_source_and_target(self, source: int, target: int) -> float:
        """
        Returns the homophily value between a source and a target node based on their class values.
        This homophily value is inferred from the mixing matrix.

        Parameters
        ----------
        source: int
            Source node id

        target: int
            Target node id

        Returns
        -------
        h: float
            homophily (similarity) between source and target nodes (minimum=0, maximum=1.)
        """
        return self.mixing_matrix[self.node_labels[source], self.node_labels[target]]

    ############################################################
    # Generation
    ############################################################

    def _initialize(self, class_attribute: str = 'm', class_values: list = None, class_labels: list = None):
        """
        Initializes the model.

        Parameters
        ----------
        class_attribute: str
            name of the attribute that represents the class

        class_values: list
            values of the class attribute

        class_labels: list
            labels of the class attribute mapping the class_values.
        """
        Graph._initialize(self, class_attribute, class_values, class_labels)
        self.h_MM = val.calibrate_null_probabilities(self.h_MM)
        self.h_mm = val.calibrate_null_probabilities(self.h_mm)
        self.mixing_matrix = np.array([[self.h_MM, 1 - self.h_MM], [1 - self.h_mm, self.h_mm]])

    def get_target_probabilities(self, source: Union[None, int], target_set: Union[None, Set[int], np.array],
                                 special_targets: Union[None, object, iter] = None) -> tuple[np.array, set[int]]:
        """
        Returns the probabilities of selecting a target node from a set of nodes based on homophily.
        Homophily is inferred from the mixing matrix.

        Parameters
        ----------
        source: int
            source node

        target_set: set[int]
            set of target nodes

        special_targets: object
            special targets

        Returns
        -------
        tuple[np.array, set[int]]
            probabilities of selecting a target node from a set of nodes, and the set of target nodes`
        """
        probs = np.array([self.get_homophily_between_source_and_target(source, target) for target in target_set])
        probs /= probs.sum()
        return probs, target_set

    def get_target(self, source: Union[None, int], targets: Union[None, Set[int]],
                   special_targets: Union[None, object, iter]) -> int:
        """
        Picks a random target node based on the homophily dynamic.

        Parameters
        ----------
        source: int
            Newly added node

        targets: Set[int]
            Potential target nodes in the graph

        special_targets: object
            Special target nodes in the graph

        Returns
        -------
            int: Target node that an edge should be added to from `source`
        """
        # Collect probabilities to connect to each node in target_list
        target_set = self.get_potential_nodes_to_connect(source, targets)
        probs = self.get_target_probabilities(source, target_set, special_targets)
        return np.random.choice(a=target_set, size=1, replace=False, p=probs)[0]

    ############################################################
    # Calculations
    ############################################################

    def info_params(self):
        """
        Shows the parameters of the model.
        """
        print('h_MM: {}'.format(self.h_MM))
        print('h_mm: {}'.format(self.h_mm))
        print('mixing matrix: \n{}'.format(self.mixing_matrix))

    def info_computed(self):
        """
        Shows the computed properties of the graph.
        """
        inferred_h_MM, inferred_h_mm = self.infer_homophily_values()
        print("- Empirical homophily within majority: {}".format(inferred_h_MM))
        print("- Empirical homophily within minority: {}".format(inferred_h_mm))

    def infer_homophily_values(self) -> Tuple[float, float]:
        """
        Infers analytically the homophily values for the majority and minority classes.

        Returns
        -------
        h_MM: float
            homophily within majority group

        h_mm: float
            homophily within minority group
        """

        e = self.calculate_edge_type_counts()
        if self.is_directed():
            h_MM = e['MM'] / (e['MM'] + e['Mm'])
            h_mm = e['mm'] / (e['mm'] + e['mM'])
        else:
            h_MM = e['MM'] / (e['MM'] + e['Mm'] + e['mM'])
            h_mm = e['mm'] / (e['mm'] + e['mM'] + e['Mm'])

        return h_MM, h_mm
