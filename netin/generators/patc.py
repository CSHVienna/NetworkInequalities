from collections import defaultdict
from typing import Union, Set

import numpy as np

from netin.utils import constants as const
from netin.utils import validator as val
from .graph import Graph


class PATC(Graph):

    ############################################################
    # Constructor
    ############################################################

    def __init__(self, n: int, k: int, f_m: float, tc: float, seed: object = None, **attr: object):
        """

        Parameters
        ----------
        n: int
            number of nodes (minimum=2)

        k: int
            minimum degree of nodes (minimum=1)

        f_m: float
            fraction of minorities (minimum=1/n, maximum=(n-1)/n)

        tc: float
            probability of a new edge to close a triad (minimum=0, maximum=1.)

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
        Graph.__init__(self, n, k, f_m, seed, **attr)
        self.tc = tc

    ############################################################
    # Init
    ############################################################

    def _infer_model_name(self):
        """
        Infers the name of the model.
        """
        return self.set_model_name(const.PATC_MODEL_NAME)

    def _validate_parameters(self):
        """
        Validates the parameters of the graph.
        """
        super()._validate_parameters()
        val.validate_float(self.tc, minimum=0., maximum=1., allow_none=True)

    def get_metadata_as_dict(self) -> dict:
        obj = super().get_metadata_as_dict()
        obj.update({
            'tc': self.tc
        })
        return obj

    ############################################################
    # Getters & Setters
    ############################################################

    def set_triadic_closure_prob(self, tc):
        """
        Parameters
        ----------
        tc: float
            probability of a new edge to close a triad (minimum=0, maximum=1.)
        """
        self.tc = tc

    def get_triadic_closure_prob(self):
        """
        Returns
        -------
        tc: float
            probability of a new edge to close a triad (minimum=0, maximum=1.)
        """
        return self.tc

    ############################################################
    # Generation
    ############################################################

    def get_special_targets(self, source: int) -> object:
        """
        Return an empty dictionary (source node ids)
        Parameters
        ----------
        source: int
            Newly added node
        """
        return defaultdict(int)

    def get_target_by_triadic_closure(self, source: Union[None, int], targets: Union[None, Set[int]],
                                      special_targets: Union[None, object, iter]) -> int:
        # Pick a target node based on triadic closure
        target_list, probs = zip(*[(t, w) for t, w in special_targets.items()])
        probs = np.array(probs)
        target = np.random.choice(
            a=target_list,  # Nodes themselves
            p=probs / probs.sum(),  # Weight by frequency
            size=1,
            replace=False,
        )[0]  # Select k=1 target
        return target

    def get_target_regular(self, source: int, targets: Set[int], special_targets=None) -> int:
        return self.get_target_by_preferential_attachment(source, targets, special_targets)

    def get_target(self, source: Union[None, int], targets: Union[None, Set[int]],
                   special_targets: Union[None, object, iter]) -> int:
        """
        Picks a random target node based on the homophily/preferential attachment dynamic.

        Parameters
        ----------
        special_targets: dict
            Dictionary of target candidates (of a given source node)

        source: None

        targets: None

        Returns
        -------
            int: Target node that an edge should be added to
        """
        # TODO: Find a better way as the conversion takes O(N)
        tc_prob = np.random.random()

        if tc_prob < self.tc and len(special_targets) > 0:
            target = self.get_target_by_triadic_closure(source, targets, special_targets)
        else:
            # Pick a target node based on preferential attachment
            target = self.get_target_regular(source, targets, special_targets)

        return target

    def update_special_targets(self, idx_target: int, source: int, target: int, targets: Set[int],
                               special_targets: object):
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
        return None
