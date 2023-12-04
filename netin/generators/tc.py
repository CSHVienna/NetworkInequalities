from collections import defaultdict
from typing import Union
from typing import List, Any, Tuple, Dict

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

    seed: object
        seed for random number generator

    Notes
    -----
    This class does not generate a graph.
    """

    ############################################################
    # Constructor
    ############################################################

    def __init__(self, n: int, f_m: float, tc: float, seed: object = None):
        Graph.__init__(self, n=n, f_m=f_m, seed=seed)
        self.tc = tc
        self.model_name = const.TC_MODEL_NAME

    ############################################################
    # Init
    ############################################################

    def validate_parameters(self):
        """
        Validates the parameters of the undirected.
        """
        Graph.validate_parameters(self)
        val.validate_float(self.tc, minimum=0., maximum=1.)

    def get_metadata_as_dict(self) -> Dict[str, Any]:
        """
        Returns the metadata (parameters) of the model as a dictionary.

        Returns
        -------
        dict
            metadata of the model
        """
        obj = Graph.get_metadata_as_dict(self)
        obj.update({
            'tc': self.tc
        })
        return obj

    ############################################################
    # Getters & Setters
    ############################################################

    def set_triadic_closure(self, tc: float):
        """
        Sets the triadic closure probability `tc`.

        Parameters
        ----------
        tc: float
            triadic closure probability (minimum=0, maximum=1)
        """
        assert 0. <= tc <= 1.,\
               f"Triadic closure probability should be between 0. and 1. but is {tc}"
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

    def initialize(self,
                   class_attribute: str = 'm',
                   class_values: List[Any] = None,
                   class_labels: List[str] = None):
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
        Graph.initialize(self, class_attribute, class_values, class_labels)

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

    def get_target_probabilities(self, source: int,
                                 available_nodes: List[int],
                                 special_targets: Union[None, Dict[int, float]] = None) -> Tuple[np.array, List[int]]:
        """Returns the probabilities of selecting a target node from a set of nodes based on triadic closure, or a regular mechanism,

        Parameters
        ----------
        source : int
            source node
        available_nodes : List[int]
            list of available target nodes
        special_targets : Union[None, Dict[int, float]], optional
            List of limited target nodes, by default None

        Returns
        -------
        Tuple[np.array, List[int]]
            Tuple of two equally sizes lists.
            The first list contains the probabilities and the second list the available nodes.
        """
        # Triadic closure is not uniform (biased towards common neighbors)
        available_nodes, probs = zip(*list(special_targets.items()))
        probs = np.array(probs).astype(np.float32)
        probs /= probs.sum()
        return probs, available_nodes

    def update_special_targets(self,
                               idx_target: int,
                               source: int, target: int,
                               available_nodes: List[int],
                               special_targets: Union[None, Dict[int, int]]) -> Union[None, Dict[int, int]]:
        """
        Updates the set of special available_nodes based on the triadic closure mechanism.
        When an edge is created, multiple potential triadic closures emerge (i.e., two-hop neighbors that are not yet
        directly connected). These are added to the set of special available_nodes.

        Parameters
        ----------
        idx_target: int
            index of the target node

        source: int
            source node

        target: int
            target node

        available_nodes: List[int]
            list of target nodes

        special_targets: Union[None, Dict[int, int]]
            special available_nodes

        Returns
        -------
         Union[None, Dict[int, int]
            updated special available_nodes
        """
        if idx_target < self.k - 1:
            # Remove target candidates of source
            available_nodes.remove(target)
            if target in special_targets:
                del(special_targets[target])  # Remove target from TC candidates

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
        """
        Shows the parameters of the model.
        """
        print('tc: {}'.format(self.tc))

    def info_computed(self):
        """
        Shows the computed properties of the graph.
        """
        inferred_tc = self.infer_triadic_closure()
        print("- Empirical triadic closure: {}".format(inferred_tc))

    def infer_triadic_closure(self) -> float:
        """
        Infers analytically the triadic closure value of the graph.

        Returns
        -------
        float
            triadic closure probability of the graph
        """
        # @TODO: To be implemented
        raise NotImplementedError("Inferring triadic closure not implemented yet.")
