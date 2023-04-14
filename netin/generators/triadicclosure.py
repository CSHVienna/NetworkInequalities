from collections import defaultdict
from typing import Union, Set

import numpy as np

from netin.generators.graph import Graph
from netin.utils import constants as const
from netin.utils import validator as val


class TriadicClosure(Graph):

    ############################################################
    # Constructor
    ############################################################

    def __init__(self, n: int, f_m: float, tc: float, seed: object = None):
        """

        Parameters
        ----------
        n: int
            number of nodes (minimum=2)

        f_m: float
            fraction of minorities (minimum=1/n, maximum=(n-1)/n)

        tc: float
            triadic closure probability (minimum=0, maximum=1)

        seed: object
            seed for random number generator

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
        self.tc = tc

    ############################################################
    # Init
    ############################################################

    def _infer_model_name(self):
        """
        Infers the name of the model.
        """
        return self.set_model_name(const.TC_MODEL_NAME)

    def _validate_parameters(self):
        """
        Validates the parameters of the undirected.
        """
        Graph._validate_parameters(self)
        val.validate_float(self.tc, minimum=0., maximum=1.)

    def get_metadata_as_dict(self) -> dict:
        obj = Graph.get_metadata_as_dict(self)
        obj.update({
            'tc': self.tc,
        })
        return obj

    ############################################################
    # Getters & Setters
    ############################################################

    def set_triadic_closure(self, tc):
        """
        Parameters
        ----------
        tc: float
            triadic closure probability (minimum=0, maximum=1)
        """
        self.tc = tc

    def get_triadic_closure(self):
        """
        Returns
        -------
        tc: float
            triadic closure probability (minimum=0, maximum=1)
        """
        return self.tc

    ############################################################
    # Generation
    ############################################################

    def _initialize(self, class_attribute: str = 'm', class_values: list = None, class_labels: list = None):
        Graph._initialize(self, class_attribute, class_values, class_labels)

    def get_special_targets(self, source: int) -> object:
        """
        Return an empty dictionary (source node ids)
        Parameters
        ----------
        source: int
            Newly added node
        """
        return defaultdict(int)

    def get_target_probabilities_regular(self, source: Union[None, int], target_set: Union[None, Set[int]],
                                         special_targets: Union[None, object, iter] = None) -> tuple[
        np.array, set[int]]:
        # return Graph.get_target_probabilities(self, source, target_set, special_targets)
        pass

    def get_target_probabilities(self, source: Union[None, int], target_set: Union[None, Set[int]],
                                 special_targets: Union[None, object, iter] = None) -> tuple[np.array, set[int]]:
        tc_prob = np.random.random()

        if tc_prob < self.tc and len(special_targets) > 0:
            target_set, probs = zip(*[(t, w) for t, w in special_targets.items()])
            probs = np.array(probs).astype(np.float32)
            probs /= probs.sum()
            target_set = set(target_set)
            return probs, target_set

        # Pick a target node based on preferential attachment
        return self.get_target_probabilities_regular(source, target_set, special_targets)

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

    def update_special_targets(self, idx_target: int, source: int, target: int, targets: Set[int],
                               special_targets: object) -> object:
        if idx_target < self.k - 1:
            # Remove target candidates of source
            targets.discard(target)
            if target in special_targets:
                del special_targets[target]  # Remove target from TC candidates

            # Incr. occurrence counter for friends of new friend
            for neighbor in self.neighbors(target):
                # G[source] gives direct access (O(1)) to source's neighbors
                # G.neighbors(source) returns an iterator which would
                # need to be searched iteratively
                if neighbor not in self[source]:
                    special_targets[neighbor] += 1
        return special_targets

    ############################################################
    # Calculations
    ############################################################

    def info_params(self):
        print('tc: {}'.format(self.tc))

    def info_computed(self):
        inferred_tc = self.infer_triadic_closure()
        print("- Empirical triadic closure: {}".format(inferred_tc))

    def infer_triadic_closure(self) -> float:
        # @TODO: To be implemented
        tc = None
        return tc
