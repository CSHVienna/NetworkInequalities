from collections import defaultdict
from typing import Union, Set, Tuple

import networkx as nx
import numpy as np
import powerlaw

from netin.utils import constants as const
from netin.utils import validator as val
from .graph import Graph


class DiGraph(nx.DiGraph, Graph):
    """Directed graph base model.

    Parameters
    ----------
    n: int
        number of nodes (minimum=2)

    d: float
        edge density (minimum=0, maximum=1)

    f_m: float
        fraction of minorities (minimum=1/n, maximum=(n-1)/n)

    plo_M: float
        activity (out-degree power law exponent) majority group (minimum=1)

    plo_m: float
        activity (out-degree power law exponent) minority group (minimum=1)

    seed: object
        seed for random number generator

    Notes
    -----
    The initialization is a directed graph with n nodes and no edges.
    Source nodes are selected based on their activity given by plo_M (if majority) or plo_m (if minority).
    Target nodes are selected depending on the chosen mechanism of edge formation.

    - DPAH: preferential attachment (in-degree) and homophily (h**), see :class:`netin.DPAH`
    - DPA: preferential attachment (in-degree), see :class:`netin.DPA`
    - DH: homophily (h**), see :class:`netin.DH`

    References
    ----------
    .. [Espin-Noboa2022] L. Espín-Noboa, C. Wagner, M. Strohmaier, & F. Karimi "Inequality and inequity in network-based ranking and recommendation algorithms" Scientific reports 12(1), 1-14, 2022.
    .. [Karimi2018] F. Karimi, M. Génois, C. Wagner, P. Singer, & M. Strohmaier, M "Homophily influences ranking of minorities in social networks", Scientific reports 8(1), 11077, 2018.
    .. [BarabasiAlbert1999] A. L. Barabasi and R. Albert "Emergence of scaling in random networks", Science 286, pp 509-512, 1999.
    """

    ############################################################
    # Constructor
    ############################################################

    def __init__(self, n: int, f_m: float, d: float, plo_M: float, plo_m: float, seed: object = None):
        nx.DiGraph.__init__(self)
        Graph.__init__(self, n=n, f_m=f_m, seed=seed)
        self.d = d
        self.plo_M = plo_M
        self.plo_m = plo_m
        self.in_degrees = None
        self.out_degrees = None
        self.activity = None
        self.expected_number_of_edges = None

    ############################################################
    # Init
    ############################################################

    def _infer_model_name(self):
        """
        Infers the name of the model.
        """
        return self.set_model_name(const.DIRECTED_MODEL_NAME)

    def _validate_parameters(self):
        """
        Validates the parameters of the directed.
        """
        Graph._validate_parameters(self)
        val.validate_float(self.d, minimum=1. / (self.n * (self.n - 1)), maximum=1.)
        val.validate_float(self.plo_M, minimum=1. + const.EPSILON)
        val.validate_float(self.plo_m, minimum=1. + const.EPSILON)

    def get_metadata_as_dict(self) -> dict:
        """
        Returns metadata for a directed.
        """
        obj = super().get_metadata_as_dict()
        obj.update({
            'd': self.d,
            'plo_M': self.plo_M,
            'plo_m': self.plo_m,
        })
        return obj

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
        self._init_edges()
        self._init_activity()

    def _init_edges(self):
        """
        Initializes the expected number of edges based on the number of nodes and density of the graph (input param).
        It also initializes the in- and out-degrees of the nodes.
        """
        self.expected_number_of_edges = int(round(self.d * self.n * (self.n - 1)))
        self.in_degrees = np.zeros(self.n)
        self.out_degrees = np.zeros(self.n)

    def _init_activity(self):
        """
        Initializes the level of activity for each node based on the power law exponents (input param).
        """
        act_M = powerlaw.Power_Law(parameters=[self.plo_M], discrete=True).generate_random(self.n_M)
        act_m = powerlaw.Power_Law(parameters=[self.plo_m], discrete=True).generate_random(self.n_m)
        self.activity = np.append(act_M, act_m)
        if np.inf in self.activity:
            self.activity[self.activity == np.inf] = 0.0
            self.activity += 1
        self.activity /= self.activity.sum()

    def get_sources(self) -> np.array:
        """
        Returns a random sample with replacement of nodes to be used as source nodes.
        The sample has the length of the expected number of edges, and the probability of each node to be selected is
        based on its activity.

        Returns
        -------
        np.array
            array of source nodes
        """
        return np.random.choice(a=np.arange(self.n), size=self.expected_number_of_edges, replace=True, p=self.activity)

    def get_target_probabilities(self, source: Union[None, int], target_set: Union[None, Set[int], np.array],
                                 special_targets: Union[None, object, iter] = None) -> np.array:
        pass

    def get_target(self, source: int, edge_list: dict, **kwargs) -> Union[None, int]:
        """
        Returns a target node for a given source node.

        Parameters
        ----------
        source: int
            source node

        edge_list: dict
            dictionary of edges

        kwargs: dict
            additional parameters

        Returns
        -------
        Union[None, int]
            target node

        Notes
        -----
        The target node must have out_degree > 0 (the older the node in the network, the more likely to get more links)
        """
        one_percent = self.n * 1 / 100.
        if np.count_nonzero(self.out_degrees) > one_percent:
            # if there are enough edges, then select only nodes with out_degree > 0 that are not already
            # connected to the source.
            # Having out_degree > 0 means they are nodes that have been in the network for at least one time step
            targets = [n for n in np.arange(self.n) if n not in edge_list[source] and self.out_degrees[n] > 0]
        else:
            # if there are no enough edges, then select all nodes that are not already connected to the source.
            targets = [n for n in np.arange(self.n) if n not in edge_list[source]]
        targets = np.delete(targets, np.where(targets == source))

        if targets.shape[0] == 0:
            return None

        probs = self.get_target_probabilities(source, targets, **kwargs)
        return np.random.choice(a=targets, size=1, replace=False, p=probs)[0]

    def generate(self):
        """
        A directed graph of n nodes is grown by attaching new nodes.
        Source nodes are selected randomly with replacement based on their activity.
        Each target node drawn based on the chosen mechanism of edge formation.

        - DPA: A graph with h_mm = h_MM in [0.5, None] is a directed BA preferential attachment model, see :class:`netin.DPA`.
        - DH: A graph with h_mm not in [0.5, None] and h_MM not in [0.5, None] is a directed Erdos-Renyi with homophily, see :class:`netin.DPH`.
        - DPAH: A graph with h_mm not in [0.5, None] and h_MM not in [0.5, None] is a DPA model with homophily, see :class:`netin.DPAH`.
        """
        # 1. Init directed and nodes (assign class labels)
        Graph.generate(self)

        # 2. Iterate until reaching desired number of edges (edge density)
        tries = 0
        edge_list = defaultdict(list)
        while self.number_of_edges() < self.expected_number_of_edges:
            tries += 1
            for source in self.get_sources():
                target = self.get_target(source, edge_list)

                if target is None:
                    continue

                if not self.has_edge(source, target):
                    self.add_edge(source, target)
                    self.in_degrees[target] += 1
                    self.out_degrees[source] += 1
                    edge_list[source].append(target)

                if self.number_of_edges() >= self.expected_number_of_edges:
                    break

            # if no more edges can be added, break
            if tries > const.MAX_TRIES_EDGE and self.number_of_edges() < self.expected_number_of_edges:
                print(f">> Edge density ({nx.density(self)}) might differ from {self.d:.5f} (n={self.n}, f_m={self.f_m}"
                      f"seed={self.seed}, plo_M={self.plo_M}, plo_m={self.plo_m}")
                break

        self._terminate()

    ############################################################
    # Calculations
    ############################################################

    def info_params(self):
        """
        Shows the (input) parameters of the graph.
        """
        print(f'd: {self.d} (expected edges: {self.expected_number_of_edges})')
        print(f'plo_M: {self.plo_M}')
        print(f'plo_m: {self.plo_m}')

    def info_computed(self):
        """
        Shows the computer properties of the graph.
        """
        for metric in ['in_degree', 'out_degree']:
            fit_M, fit_m = self.fit_powerlaw(metric)
            print(f"- Powerlaw fit ({metric}):")
            print(f"- {self.get_majority_label()}: alpha={fit_M.power_law.alpha}, sigma={fit_M.power_law.sigma}, "
                  f"min={fit_M.power_law.xmin}, max={fit_M.power_law.xmax}")
            print(f"- {self.get_minority_label()}: alpha={fit_m.power_law.alpha}, sigma={fit_m.power_law.sigma}, "
                  f"min={fit_m.power_law.xmin}, max={fit_m.power_law.xmax}")

    def calculate_in_degree_powerlaw_exponents(self) -> Tuple[float, float]:
        """
        Returns the power law exponents for the in-degree distribution of the majority and minority class.

        Returns
        -------
        Tuple[float, float]
            power law exponents for the in-degree distribution of the majority and minority class
        """
        pl_M, pl_m = self.calculate_powerlaw_exponents(metric='in_degree')

        # fit_M, fit_m = self.fit_powerlaw(metric='in_degree')
        # pl_M = fit_M.power_law.alpha
        # pl_m = fit_m.power_law.alpha

        return pl_M, pl_m

    def calculate_out_degree_powerlaw_exponents(self) -> Tuple[float, float]:
        """
        Returns the power law exponents for the out-degree distribution of the majority and minority class.

        Returns
        -------
        Tuple[float, float]
            power law exponents for the out-degree distribution of the majority and minority class
        """
        pl_M, pl_m = self.calculate_powerlaw_exponents(metric='out_degree')

        # fit_M, fit_m = self.fit_powerlaw(metric='out_degree')
        # pl_M = fit_M.power_law.alpha
        # pl_m = fit_m.power_law.alpha

        return pl_M, pl_m

    ############################################################
    # Getters and setters
    ############################################################

    def get_expected_number_of_edges(self) -> int:
        """
        Returns the expected number of edges based on number of nodes and edge density.

        Returns
        -------
        int
            expected number of edges
        """
        return self.expected_number_of_edges

    def get_expected_density(self) -> float:
        """
        Returns the expected edge density (d, the input parameter).

        Returns
        -------
        float
            expected edge density
        """
        return self.d

    def get_expected_powerlaw_out_degree_majority(self) -> float:
        """
        Returns the expected power law exponent for the out-degree distribution of the majority class
        (plo_M, the input parameter).

        Returns
        -------
        float
            expected power law exponent for the out-degree distribution of the majority class
        """
        return self.plo_M

    def get_expected_powerlaw_out_degree_minority(self):
        """
        Returns the expected power law exponent for the out-degree distribution of the minority class
        (plo_m, the input parameter).

        Returns
        -------
        float
            expected power law exponent for the out-degree distribution of the minority class
        """
        return self.plo_m

    def get_activity_distribution(self) -> np.array:
        """
        Returns the activity distribution of all the nodes in the graph.

        Returns
        -------
        np.array
            activity distribution of all the nodes in the graph
        """
        return self.activity

    def _makecopy(self):
        """
        Makes a copy of the current object.
        """
        obj = self.__class__(n=self.n,
                             d=self.d,
                             f_m=self.f_m,
                             plo_M=self.plo_M,
                             plo_m=self.plo_m,
                             seed=self.seed)
        return obj

