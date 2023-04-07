from collections import defaultdict
from typing import Callable, Union, Set

import networkx as nx
import numpy as np

from tmp import metadata as md
from netin.utils import constants as const


class Graph(nx.Graph, md.Metadata):

    def __init__(self, n: int, k: int, f_m: float, h_MM: float = None, h_mm: float = None, tc: float = None, seed: object = None,
                 _on_edge_added: Union[None, Callable[[nx.Graph, int, int], None]] = None,
                 _on_tc_edge_added: Union[None, Callable[[nx.Graph, int, int], None]] = None, **attr: object):
        """
        Initializes a random undigraph using the BA preferential attachment model [1].
        This class accounts for added homophily and triadic closure.

        Parameters
        ----------
        n: int
            number of nodes (minimum=2)

        k: int
            minimum degree of nodes (minimum=1)

        f_m: float
            fraction of minorities (minimum=1/n, maximum=(n-1)/n)

        h_MM: float
            homophily (similarity) between majority nodes (minimum=0, maximum=1.)

        h_mm: float
            homophily (similarity) between minority nodes (minimum=0, maximum=1.)

        tc: float
            probability of a new edge to close a triad (minimum=0, maximum=1.)

        seed: float, optional
            seed for random number generator (default=None).

        _on_edge_added: Union[None, Callable[[nx.Graph, int, int], None] (default=None)
            callback function to be called when an edge is added
            The function is expected to take the undigraph and two ints (source and target node) as input.

        _on_tc_edge_added: Union[None, Callable[[nx.Graph, int, int], None] (default=None)
            callback function to be called when an edge is added by triadic closure
            The function is expected to take the undigraph and two ints (source and target node) as input.

        attr: dict
            attributes to add to undigraph as key=value pairs

        Notes
        -----
        The initialization is a undigraph with n nodes and no edges.
        Then, everytime a node is selected as source, it gets connected to k target nodes.
        Target nodes are selected via preferential attachment (in-degree), homophily (h_**),
        and/or triadic closure (tc).

        References
        ----------
        - [1] A. L. Barabasi and R. Albert "Emergence of scaling in random networks", Science 286, pp 509-512, 1999.
        """
        super().__init__(**attr)
        md.Metadata.__init__(self, n=n, k=k, f_m=f_m, h_MM=h_MM, h_mm=h_mm, tc=tc, seed=seed)
        self._on_edge_added = _on_edge_added
        self._on_tc_edge_added = _on_tc_edge_added
        self._validate_parameters()

    def _initialize(self, class_attribute: str = 'm', class_values: list = None, class_labels: list = None):
        """
        Initializes the random seed and the undigraph metadata.
        """
        np.random.seed(self.seed)
        self._infer_model_name()
        self._set_class_info(class_attribute, class_values, class_labels)
        self.graph = self.get_metadata_as_dict()

    def generate(self):
        """
        A undigraph of n nodes is grown by attaching new nodes each with k edges.
        Each edge is either drawn by preferential attachment, homophily, and/or triadic closure.

        For triadic closure, a candidate is chosen uniformly at random from all triad-closing edges (of the new node).
        Otherwise, or if there are no triads to close, edges are connected via preferential attachment and/or homophily.

        Homophily varies ranges from 0 (heterophilic) to 1 (homophilic), where 0.5 is neutral.
        Similarly, triadic closure varies from 0 (no triadic closure) to 1 (full triadic closure).

        . PA: A undigraph with h_mm = h_MM in [0.5, None] and tc = 0 is a BA preferential attachment model.
        . PAH: A undigraph with h_mm not in [0.5, None] and h_MM not in [0.5, None] and tc = 0 is a PA model with homophily.
        . PATC: A undigraph with h_mm = h_MM in [0.5, None] and tc > 0 is a PA model with triadic closure.
        . PATCH: A undigraph with h_mm not in [0.5, None] and h_MM not in [0.5, None] and tc > 0 is a PA model
                 with homophily and triadic closure.

        """
        self._initialize()

        # 1. Init nodes (assign class labels)
        nodes, labels = self._init_nodes()
        self.add_nodes_from(nodes)
        nx.set_node_attributes(self, labels, self.class_attribute)

        # 2. Validate homophily
        self._init_mixing_matrix()

        # 3. Node to be added, init to m to start with m pre-existing, unconnected nodes
        source = self.k

        while source < self.n:  # Iterate until N nodes were added
            targets_tc = defaultdict(int)
            targets = set(range(source))
            tc_edge = False  # Remember if edge was caused by TC
            for idx_m in range(self.k):
                tc_edge = False
                tc_prob = np.random.random()

                # Choose next target based on TC or PA+H
                if self.tc not in const.NO_TRIADIC_CLOSURE and tc_prob < self.tc and len(targets_tc) > 0:
                    # TC
                    target = self._pick_target_tc(targets_tc)
                    tc_edge = True
                elif self.h_MM not in const.NO_HOMOPHILY or self.h_mm not in const.NO_HOMOPHILY:
                    # PA+H
                    target = self._pick_target_pa_h(source, targets, labels)
                else:
                    # PA only
                    target = self._pick_target_pa(source, targets)

                # [Perf.] No need to update target dicts if we are adding the last edge
                if idx_m < self.k - 1:
                    # Remove target candidates of source
                    targets.discard(target)
                    if target in targets_tc:
                        del targets_tc[target]  # Remove target from TC candidates

                    # Incr. occurrence counter for friends of new friend
                    for neighbor in self.neighbors(target):
                        # G[source] gives direct access (O(1)) to source's neighbors
                        # G.neighbors(source) returns an iterator which would
                        # need to be searched iteratively
                        if neighbor not in self[source]:
                            targets_tc[neighbor] += 1

                # Finally add edge to undigraph
                self.add_edge(source, target)

                # Call event handlers if present
                if self._on_edge_added is not None:
                    self._on_edge_added(self, source, target)
                if self._on_tc_edge_added is not None and tc_edge:
                    self._on_tc_edge_added(self, source, target)

            # Activate node as potential target and select next node
            source += 1

        return

    def _pick_target_tc(self, targets_tc: dict) -> float:
        """
        Sample TC target based on its occurrence frequency

        Parameters
        ----------
        targets_tc: dict
            Dictionary of target nodes and their occurrence counts

        Returns
        -------
        int: Target node that an edge should be added to
        """
        # TODO: Find a better way as the conversion takes O(N)
        target_list, probs = zip(*[(t, w) for t, w in targets_tc.items()])
        probs = np.array(probs)
        target = np.random.choice(
            a=target_list,  # Nodes themselves
            p=probs / probs.sum(),  # Weight by frequency
            size=1,
            replace=False,
        )[0]  # Select k=1 target
        return target

    def _pick_target_pa_h(self, source: int, target_set: Set[int], labels: Set[bool]):
        """
        Picks a random target node based on the homophily/preferential attachment dynamic.

        Parameters
        ----------
        source: int
            Newly added node

        target_set: Set[int]
            Potential target nodes in the undigraph

        labels: Set[bool]
            Class labels of the nodes in the undigraph

        Returns
        -------
            int: Target node that an edge should be added to
        """
        # Collect probabilities to connect to each node in target_list
        target_list = [t for t in target_set if t != source and t not in nx.neighbors(self, source)]
        probs = np.array([self.mixing_matrix[labels[source], labels[target]] * (self.degree(target) + const.EPSILON)
                          for target in target_list])
        probs /= probs.sum()
        return np.random.choice(a=target_list, size=1, replace=False, p=probs)[0]

    def _pick_target_pa(self, source: int, target_set: Set[int]):
        """
        Picks a random target node based on the homophily/preferential attachment dynamic.

        Parameters
        ----------
        source: int
            Newly added node

        target_set: Set[int]
            Potential target nodes in the undigraph

        Returns
        -------
            int: Target node that an edge should be added to
        """
        # Collect probabilities to connect to each node in target_list
        target_list = [t for t in target_set if t != source and t not in nx.neighbors(self, source)]
        probs = np.array([self.degree(target) + const.EPSILON for target in target_list])
        probs /= probs.sum()
        return np.random.choice(a=target_list, size=1, replace=False, p=probs)[0]

    ############################################################
    # Calculations
    ############################################################

    def info(self):
        # super().info()
        print('Model: {}'.format(self.get_model_name()))
        print('Class attribute: {}'.format(self.get_class_attribute()))
        print('Class values: {}'.format(self.get_class_values()))
        print('Class labels: {}'.format(self.get_class_labels()))
        print('n: {}'.format(self.n))
        print('k: {}'.format(self.k))
        print('f_m: {}'.format(self.f_m))
        print('h_MM: {}'.format(self.h_MM))
        print('h_mm: {}'.format(self.h_mm))
        print('tc: {}'.format(self.tc))
        print('seed: {}'.format(self.seed))
        print(f'- minimum degree: {self.calculate_minimum_degree()}')
        print(f'- fraction of minority: {self.calculate_fraction_of_minority()}')

    def calculate_minimum_degree(self):
        return min([d for n, d in self.degree])

    def calculate_fraction_of_minority(self):
        return sum([1 for n, obj in self.nodes(data=True) if obj[self.class_attribute] == self.class_values[
            self.class_labels.index(const.MINORITY_LABEL)]]) / self.number_of_nodes()
