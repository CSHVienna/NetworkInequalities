from collections import defaultdict
from typing import Union, Set

import networkx as nx
import numpy as np
import powerlaw

from netin.utils import constants as const
from netin.utils import validator as val
from .graph import Graph


class DiGraph(nx.DiGraph, Graph):

    ############################################################
    # Constructor
    ############################################################

    def __init__(self, n: int, f_m: float, d: float, plo_M: float, plo_m: float, seed: object = None):
        """

        Parameters
        ----------
        n: int
            number of nodes (minimum=2)

        k: int
            minimum degree of nodes (minimum=1)

        f_m: float
            fraction of minorities (minimum=1/n, maximum=(n-1)/n)

        d: float
            edge density (minimum=0, maximum=1)

        plo_M: float
            activity (out-degree power law exponent) majority group (minimum=1)

        plo_m: float
            activity (out-degree power law exponent) minority group (minimum=1)

        seed: object
            seed for random number generator

        Notes
        -----
        The initialization is a directed with n nodes and no edges.
        Then, everytime a node is selected as source, it gets connected to k target nodes.
        Target nodes are selected via preferential attachment (in-degree), homophily (h_**),
        and/or triadic closure (tc).

        References
        ----------
        - [1] A. L. Barabasi and R. Albert "Emergence of scaling in random networks", Science 286, pp 509-512, 1999.
        """
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
        Graph._initialize(self, class_attribute, class_values, class_labels)
        self._init_edges()
        self._init_activity()

    def _init_edges(self):
        self.expected_number_of_edges = int(round(self.d * self.n * (self.n - 1)))
        self.in_degrees = np.zeros(self.n)
        self.out_degrees = np.zeros(self.n)

    def _init_activity(self):
        act_M = powerlaw.Power_Law(parameters=[self.plo_M], discrete=True).generate_random(self.n_M)
        act_m = powerlaw.Power_Law(parameters=[self.plo_m], discrete=True).generate_random(self.n_m)
        self.activity = np.append(act_M, act_m)
        if np.inf in self.activity:
            self.activity[self.activity == np.inf] = 0.0
            self.activity += 1
        self.activity /= self.activity.sum()

    def get_sources(self) -> np.array:
        return np.random.choice(a=np.arange(self.n), size=self.expected_number_of_edges, replace=True, p=self.activity)

    def get_target_probabilities(self, source: Union[None, int], target_set: Union[None, Set[int], np.array],
                                 special_targets: Union[None, object, iter] = None) -> np.array:
        pass

    def get_target(self, source: int, edge_list: dict, **kwargs) -> Union[None, int]:
        one_percent = self.n * 1 / 100.
        if np.count_nonzero(self.out_degrees) > one_percent:
            targets = [n for n in np.arange(self.n) if n not in edge_list[source]]
        else:
            targets = np.arange(self.n)
        targets = np.delete(targets, np.where(targets == source))

        if targets.shape[0] == 0:
            return None

        probs = self.get_target_probabilities(source, targets, **kwargs)
        return np.random.choice(a=targets, size=1, replace=False, p=probs)[0]

    def generate(self):
        """
        A directed graph of n nodes is grown by attaching new nodes.
        Each edge is either drawn by preferential attachment, homophily, or both

        Homophily varies ranges from 0 (heterophilic) to 1 (homophilic), where 0.5 is neutral.
        Similarly, triadic closure varies from 0 (no triadic closure) to 1 (full triadic closure).

        . DPA: A graph with h_mm = h_MM in [0.5, None] is a directed BA preferential attachment model.
        . DH: A graph with h_mm not in [0.5, None] and h_MM not in [0.5, None] is a directed Erdos-Renyi with homophily.
        . DPAH: A graph with h_mm not in [0.5, None] and h_MM not in [0.5, None] is a DPA model with homophily.

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
        print(f'd: {self.d} (expected edges: {self.expected_number_of_edges})')
        print(f'plo_M: {self.plo_M}')
        print(f'plo_m: {self.plo_m}')

    def info_computed(self):
        for c in self.class_values:
            fit = powerlaw.Fit(data=[d for n, d in self.out_degree() if self.nodes[n][self.class_attribute] == c],
                               discrete=True)
            print(f"- {self.class_labels[c]}: alpha={fit.power_law.alpha}, sigma={fit.power_law.sigma}, "
                  f"min={fit.power_law.xmin}, max={fit.power_law.xmax}")

    def calculate_in_degree_powerlaw_exponents(self) -> (float, float):
        vM = self.get_majority_value()
        dM = [d for n, d in self.in_degree() if self.nodes[n][self.class_attribute] == vM]
        dm = [d for n, d in self.in_degree() if self.nodes[n][self.class_attribute] != vM]

        fit_M = powerlaw.Fit(data=dM, discrete=True, xmin=min(dM), xmax=max(dM))
        fit_m = powerlaw.Fit(data=dm, discrete=True, xmin=min(dm), xmax=max(dm))

        pl_M = fit_M.power_law.alpha
        pl_m = fit_m.power_law.alpha
        return pl_M, pl_m

    def calculate_out_degree_powerlaw_exponents(self) -> (float, float):
        vM = self.get_majority_value()
        dM = [d for n, d in self.out_degree() if self.nodes[n][self.class_attribute] == vM]
        dm = [d for n, d in self.out_degree() if self.nodes[n][self.class_attribute] != vM]

        fit_M = powerlaw.Fit(data=dM, discrete=True, xmin=min(dM), xmax=max(dM))
        fit_m = powerlaw.Fit(data=dm, discrete=True, xmin=min(dm), xmax=max(dm))

        pl_M = fit_M.power_law.alpha
        pl_m = fit_m.power_law.alpha
        return pl_M, pl_m

    ############################################################
    # Getters and setters
    ############################################################

    def get_expected_number_of_edges(self) -> int:
        return self.expected_number_of_edges

    def get_expected_density(self) -> float:
        return self.d

    def get_expected_powerlaw_out_degree_majority(self):
        return self.plo_M

    def get_expected_powerlaw_out_degree_minority(self):
        return self.plo_m

    def get_activity_distribution(self):
        return self.activity
