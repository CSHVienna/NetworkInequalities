from typing import Set
from typing import Union

import networkx as nx
import numpy as np

from netin.utils import validator as val
from .graph import Graph


class UnDiGraph(Graph):

    ############################################################
    # Constructor
    ############################################################

    def __init__(self, n: int, k: int, f_m: float, seed: object = None):
        """

        Parameters
        ----------
        n: int
            number of nodes (minimum=2)

        k: int
            minimum degree of nodes (minimum=1)

        f_m: float
            fraction of minorities (minimum=1/n, maximum=(n-1)/n)

        seed: object
            seed for random number generator

        Notes
        -----
        The initialization is an undirected graph with n nodes and no edges.
        Then, everytime a node is selected as source, it gets connected to k target nodes.
        Target nodes are selected via preferential attachment (in-degree), homophily (h_**),
        and/or triadic closure (tc).

        References
        ----------
        - [1] A. L. Barabasi and R. Albert "Emergence of scaling in random networks", Science 286, pp 509-512, 1999.
        """
        Graph.__init__(self, n=n, f_m=f_m, seed=seed)
        self.k = k

    ############################################################
    # Init
    ############################################################

    def _validate_parameters(self):
        """
        Validates the parameters of the undigraph.
        """
        super()._validate_parameters()
        val.validate_int(self.k, minimum=1)

    def get_metadata_as_dict(self) -> dict:
        """
        Returns metadata for a undigraph.
        """
        obj = super().get_metadata_as_dict()
        obj.update({
            'k': self.k,
        })
        return obj

    ############################################################
    # Generation
    ############################################################

    def get_target(self, source: Union[None, int], targets: Union[None, Set[int]],
                   special_targets: Union[None, object, iter]) -> int:
        """
        Picks a random target node based on preferential attachment.

        Parameters
        ----------
        special_targets : object
            Special targets to be considered

        source: int
            Newly added node

        targets: Set[int]
            Potential target nodes in the undigraph based on preferential attachment

        Returns
        -------
            int: Target node that an edge should be added to
        """
        # Collect probabilities to connect to each node in target_list
        target_set = set([t for t in targets if t != source and t not in nx.neighbors(self, source)])
        probs, target_set = self.get_target_probabilities(source, target_set, special_targets)
        return np.random.choice(a=list(target_set), size=1, replace=False, p=probs)[0]

    def generate(self):
        """
        An undigraph of n nodes is grown by attaching new nodes each with k edges.
        Each edge is either drawn by preferential attachment, homophily, and/or triadic closure.

        For triadic closure, a candidate is chosen uniformly at random from all triad-closing edges (of the new node).
        Otherwise, or if there are no triads to close, edges are connected via preferential attachment and/or homophily.

        Homophily varies ranges from 0 (heterophilic) to 1 (homophilic), where 0.5 is neutral.
        Similarly, triadic closure varies from 0 (no triadic closure) to 1 (full triadic closure).

        . PA:    An undirected graph with h_mm = h_MM in [0.5, None] and tc = 0 is a BA preferential attachment model.
        . PAH:   An undirected graph with h_mm not in [0.5, None] and h_MM not in [0.5, None] and tc = 0 is a PA model
                 with homophily.
        . PATC:  An undirected graph with h_mm = h_MM in [0.5, None] and tc > 0 is a PA model with triadic closure.
        . PATCH: An undirected graph with h_mm not in [0.5, None] and h_MM not in [0.5, None] and tc > 0 is a PA model
                 with homophily and triadic closure.

        """
        # 1. Init an undirected graph and nodes (assign class labels)
        super().generate()

        # 2. Iterate until n nodes are added (starts with k pre-existing, unconnected nodes)
        for source in self.node_list[self.k:]:
            targets = set(range(source))  # targets via preferential attachment
            special_targets = self.get_special_targets(source)

            for idx_target in range(self.k):
                # Choose next target
                target = self.get_target(source, targets, special_targets)

                special_targets = self.update_special_targets(idx_target, source, target, targets, special_targets)

                # Finally add edge to undigraph
                self.add_edge(source, target)

                # Call event handlers if present
                self.on_edge_added(source, target)

        self._terminate()

    ############################################################
    # Getters and Setters
    ############################################################

    def get_expected_number_of_edges(self) -> int:
        return (self.get_expected_number_of_nodes() * self.get_expected_minimum_degree()) - \
            (self.get_expected_minimum_degree() ** self.get_expected_minimum_degree())

    def get_expected_minimum_degree(self) -> int:
        return self.k

    ############################################################
    # Calculations
    ############################################################

    def info_params(self):
        print('k: {}'.format(self.k))
