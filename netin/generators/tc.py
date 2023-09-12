from collections import defaultdict
from typing import Union, Set

import numpy as np

from netin.utils import constants as const
from netin.utils import validator as val
from .graph import Graph


class TriadicClosure(Graph):
    """Class to model triadic closure as a mechanism of edge formation given a source and a target node.

    Parameters
    ----------
    n: int
        number of nodes (minimum=2)

    f_m: float
        fraction of minorities (minimum=1/n, maximum=(n-1)/n)

    tc: float
        triadic closure probability (minimum=0, maximum=1)

    tc_uniform: bool
        specifies whether the triadic closure target is chosen uniform at random or if it follows the regular link formation mechanisms (e.g., homophily) (default=True)

    seed: object
        seed for random number generator

    Notes
    -----
    This class does not generate a graph.
    """

    ############################################################
    # Constructor
    ############################################################

    def __init__(self, n: int, f_m: float, tc: float, tc_uniform: bool = False, seed: object = None):
        Graph.__init__(self, n=n, f_m=f_m, seed=seed)
        self.tc = tc
        self.tc_uniform = tc_uniform

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
        """
        Returns the metadata (parameters) of the model as a dictionary.

        Returns
        -------
        dict
            metadata of the model
        """
        obj = Graph.get_metadata_as_dict(self)
        obj.update({
            'tc': self.tc,
            'tc_uniform': self.tc_uniform
        })
        return obj

    ############################################################
    # Getters & Setters
    ############################################################

    def set_triadic_closure(self, tc):
        """
        Sets the triadic closure probability `tc`.

        Parameters
        ----------
        tc: float
            triadic closure probability (minimum=0, maximum=1)
        """
        self.tc = tc

    def get_triadic_closure(self) -> float:
        """
        Returns the triadic closure probability `tc`.

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

    def get_special_targets(self, source: int) -> object:
        """
        Returns an empty dictionary (source node ids)

        Parameters
        ----------
        source : int
            Newly added node

        Returns
        -------
        object
            Return an empty dictionary (source node ids)
        """
        return defaultdict(int)

    def get_target_probabilities_regular(self, source: Union[None, int],
                                         target_set: Union[None, Set[int]],
                                         special_targets: Union[None, object, iter] = None) -> \
            tuple[np.array, set[int]]:
        # return Graph.get_target_probabilities(self, source, target_set, special_targets)
        pass

    def get_target_probabilities(self, source: Union[None, int], target_set: Union[None, Set[int]],
                                 special_targets: Union[None, object, iter] = None) -> tuple[np.array, set[int]]:
        """
        Returns the probabilities of selecting a target node from a set of nodes based on triadic closure,
        or a regular mechanism.

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
        tc_prob = np.random.random()

        if tc_prob < self.tc and len(special_targets) > 0:
            if not self.tc_uniform:
                return self.get_target_probabilities_regular(source, special_targets)
            target_set, probs = zip(*[(t, w) for t, w in special_targets.items()])
            probs = np.array(probs).astype(np.float32)
            probs /= probs.sum()
            target_set = set(target_set)
            return probs, target_set

        # Pick a target
        return self.get_target_probabilities_regular(source, target_set, special_targets)

    def get_target(self, source: Union[None, int],
                   targets: Union[None, Set[int]],
                   special_targets: Union[None, object, iter]) -> int:
        """
        Picks a random target node based on the homophily/preferential attachment dynamic.

        Parameters
        ----------
        source: int
            Newly added node

        targets: Set[int]
            Potential target nodes in the graph

        special_targets: object
            Special target nodes

        Returns
        -------
            int
                Target node that an edge should be created from `source`
        """
        # Collect probabilities to connect to each node in target_list
        target_set = self.get_potential_nodes_to_connect(source, targets)
        probs = self.get_target_probabilities(source, target_set, special_targets)
        return np.random.choice(a=target_set, size=1, replace=False, p=probs)[0]

    def update_special_targets(self, idx_target: int, source: int, target: int, targets: Set[int],
                               special_targets: object) -> object:
        """
        Updates the set of special targets based on the triadic closure mechanism.
        When an edge is created, multiple potential triadic closures emerge (i.e., two-hop neighbors that are not yet
        directly connected). These are added to the set of special targets.

        Parameters
        ----------
        idx_target: int
            index of the target node

        source: int
            source node

        target: int
            target node

        targets: Set[int]
            set of target nodes

        special_targets: object
            special targets

        Returns
        -------
        object
            updated special targets
        """
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
                    special_targets[neighbor] = 1
        return special_targets

    ############################################################
    # Calculations
    ############################################################

    def info_params(self):
        """
        Shows the parameters of the model.
        """
        print('tc: {}'.format(self.tc))
        print('tc_uniform: {}'.format(self.tc_uniform))

    def info_computed(self):
        """
        Shows the computed properties of the graph.
        """
        inferred_tc = self.infer_triadic_closure()
        print("- Empirical triadic closure: {}".format(inferred_tc))

    def infer_triadic_closure(self) -> float:
        # @TODO: To be implemented
        tc = None
        return tc
