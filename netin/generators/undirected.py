from typing import Set
from typing import Tuple
from typing import Union

import networkx as nx
import numpy as np
import powerlaw

from netin.utils import constants as const
from netin.utils import validator as val
from .graph import Graph


class UnDiGraph(Graph):
    """Undirected graph base model.

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
    Target nodes are selected depending on the chosen mechanism of edge formation:

    - PA: Preferential attachment (in-degree), see :class:`netin.PA` [BarabasiAlbert1999]_
    - PAH: Preferential attachment (in-degree) with homophily, see :class:`netin.PAH` [Karimi2018]_
    - PATC: Preferential attachment (in-degree) with triadic closure, see :class:`netin.PATC` [HolmeKim2002]_
    - PATCH: Preferential attachment (in-degree) with homophily and triadic closure, see :class:`netin.PATCH`

    References
    ----------
    .. [BarabasiAlbert1999] A. L. Barabasi and R. Albert "Emergence of scaling in random networks", Science 286, pp 509-512, 1999.
    .. [Karimi2018] F. Karimi, M. Génois, C. Wagner, P. Singer, & M. Strohmaier, M "Homophily influences ranking of minorities in social networks", Scientific reports 8(1), 11077, 2018.
    .. [HolmeKim2002] P. Holme and B. J. Kim “Growing scale-free networks with tunable clustering” Phys. Rev. E 2002.
    """

    ############################################################
    # Constructor
    ############################################################

    def __init__(self, n: int, k: int, f_m: float, seed: object = None):
        Graph.__init__(self, n=n, f_m=f_m, seed=seed)
        self.k = k

    ############################################################
    # Init
    ############################################################

    def _validate_parameters(self):
        """
        Validates the parameters of the undirected.
        """
        super()._validate_parameters()
        val.validate_int(self.k, minimum=1)

    def get_metadata_as_dict(self) -> dict:
        """
        Returns metadata for a undirected.
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
            Potential target nodes in the undirected based on preferential attachment

        Returns
        -------
            int
                Target node that an edge should be added to
        """
        # Collect probabilities to connect to each node in target_list
        target_set = set([t for t in targets if t != source and t not in nx.neighbors(self, source)])
        probs, target_set = self.get_target_probabilities(source, target_set, special_targets)
        return np.random.choice(a=list(target_set), size=1, replace=False, p=probs)[0]

    def generate(self):
        """
        An undirected graph of n nodes is grown by attaching new nodes each with k edges.
        Each edge is either drawn by preferential attachment, homophily, and/or triadic closure.

        For triadic closure, a candidate is chosen uniformly at random from all triad-closing edges (of the new node).
        Otherwise, or if there are no triads to close, edges are connected via preferential attachment and/or homophily.

        Homophily varies ranges from 0 (heterophilic) to 1 (homophilic), where 0.5 is neutral.
        Similarly, triadic closure varies from 0 (no triadic closure) to 1 (full triadic closure).

        - PA: An undirected graph with h_mm = h_MM in [0.5, None] and tc = 0 is a BA preferential attachment model.
        - PAH: An undirected graph with h_mm not in [0.5, None] and h_MM not in [0.5, None] and tc = 0 is a PA model with homophily.
        - PATC: An undirected graph with h_mm = h_MM in [0.5, None] and tc > 0 is a PA model with triadic closure.
        - PATCH: An undirected graph with h_mm not in [0.5, None] and h_MM not in [0.5, None] and tc > 0 is a PA model with homophily and triadic closure.

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

                # Finally add edge to undirected
                self.add_edge(source, target)

                # Call event handlers if present
                self.on_edge_added(source, target)

        self._terminate()

    ############################################################
    # Getters and Setters
    ############################################################

    def _infer_model_name(self):
        """
        Infers the name of the model.
        """
        return self.set_model_name(const.UNDIRECTED_MODEL_NAME)

    def get_expected_number_of_edges(self) -> int:
        """
        Computes and returns the expected number of edges based on minimum degree `k` and number of nodes `n`

        Returns
        -------
        int
            Expected number of edges
        """
        return (self.get_expected_number_of_nodes() * self.get_expected_minimum_degree()) - \
            (self.get_expected_minimum_degree() ** self.get_expected_minimum_degree())

    def get_expected_minimum_degree(self) -> int:
        """
        Returns the expected minimum degree of the graph (`k`, the input parameter)

        Returns
        -------
        int
            Expected minimum degree
        """

        return self.k

    ############################################################
    # Calculations
    ############################################################

    def info_params(self):
        """
        Shows the parameters of the model.
        """
        print('k: {}'.format(self.k))

    def info_computed(self):
        """
        Shows the computed properties of the graph.
        """
        fit_M, fit_m = self.fit_degree_powerlaw()
        print(f"- Powerlaw fit (degree):")
        print(f"- {self.get_majority_label()}: alpha={fit_M.power_law.alpha}, "
              f"sigma={fit_M.power_law.sigma}, "
              f"min={fit_M.power_law.xmin}, max={fit_M.power_law.xmax}")
        print(f"- {self.get_minority_label()}: alpha={fit_m.power_law.alpha}, "
              f"sigma={fit_m.power_law.sigma}, "
              f"min={fit_m.power_law.xmin}, max={fit_m.power_law.xmax}")

    def fit_degree_powerlaw(self) -> Tuple[powerlaw.Fit, powerlaw.Fit]:
        """
        Returns the powerlaw fit of the degree distribution to a powerlaw for the majority and minority class.

        Returns
        -------
        fit_M : powerlaw.Fit
            Powerlaw fit for the majority class

        fit_m: powerlaw.Fit
            Powerlaw fit for the minority class
        """
        fit_M, fit_m = self.fit_powerlaw(metric='degree')

        # vM = self.get_majority_value()
        # dM = [d for n, d in self.degree() if self.nodes[n][self.class_attribute] == vM]
        # dm = [d for n, d in self.degree() if self.nodes[n][self.class_attribute] != vM]
        #
        # fit_M = powerlaw.Fit(data=dM, discrete=True, xmin=min(dM), xmax=max(dM))
        # fit_m = powerlaw.Fit(data=dm, discrete=True, xmin=min(dm), xmax=max(dm))

        return fit_M, fit_m

    def calculate_degree_powerlaw_exponents(self) -> Tuple[float, float]:
        """
        Returns the powerlaw exponents for the majority and minority class.

        Returns
        -------
        pl_M : float
            Powerlaw exponent for the majority class

        pl_m: float
            Powerlaw exponent for the minority class
        """
        pl_M, pl_m = self.calculate_powerlaw_exponents(metric='degree')

        # pl_M, pl_m = calculate_degree_powerlaw_exponents(self)
        # fit_M, fit_m = self.fit_degree_powerlaw()
        # pl_M = fit_M.power_law.alpha
        # pl_m = fit_m.power_law.alpha

        return pl_M, pl_m

    def _makecopy(self):
        """
        Makes a copy of the current object.
        """
        return self.__class__(n=self.n,
                              k=self.k,
                              f_m=self.f_m,
                              seed=self.seed)
