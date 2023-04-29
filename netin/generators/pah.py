from typing import Union, Set, Tuple

import numpy as np
from sympy import Eq
from sympy import solve
from sympy import symbols

import netin
from netin.generators.h import Homophily
from netin.utils import constants as const
from .pa import PA


class PAH(PA, Homophily):
    """Creates a new PAH instance. An undirected graph with preferential attachment and homophily.

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

    seed: object
        seed for random number generator

    Notes
    -----
    The initialization is an undirected graph with n nodes, where f_m are the minority.
    Then, everytime a node is selected as source, it gets connected to k target nodes.
    Target nodes are selected via preferential attachment (in-degree) and homophily (h_**).
    This model is based on [Karimi2018]_ known as the "Barabasi model with homophily" or "BA Homophily".
    """

    ############################################################
    # Constructor
    ############################################################

    def __init__(self, n: int, k: int, f_m: float, h_MM: float, h_mm: float, seed: object = None):
        PA.__init__(self, n=n, k=k, f_m=f_m, seed=seed)
        Homophily.__init__(self, n=n, f_m=f_m, h_MM=h_MM, h_mm=h_mm, seed=seed)

    ############################################################
    # Init
    ############################################################

    def _infer_model_name(self):
        """
        Infers the name of the model.
        """
        return self.set_model_name(const.PAH_MODEL_NAME)

    def _validate_parameters(self):
        """
        Validates the parameters of the undirected.
        """
        PA._validate_parameters(self)
        Homophily._validate_parameters(self)

    def get_metadata_as_dict(self) -> dict:
        """
        Returns the metadata (parameters) of the model as a dictionary.

        Returns
        -------
        dict
            metadata of the model
        """
        obj = PA.get_metadata_as_dict(self)
        obj.update(Homophily.get_metadata_as_dict(self))
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
        PA._initialize(self, class_attribute, class_values, class_labels)
        Homophily._initialize(self, class_attribute, class_values, class_labels)

    def get_target_probabilities(self, source: Union[None, int], target_set: Union[None, Set[int]],
                                 special_targets: Union[None, object, iter] = None) -> tuple[np.array, set[int]]:
        """
        Returns the probabilities of selecting a target node from a set of nodes based on the preferential attachment
        and homophily.

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
            probabilities of selecting a target node from a set of nodes, and the set of target nodes
        """
        probs = np.array([self.get_homophily_between_source_and_target(source, target) *
                          (self.degree(target) + const.EPSILON) for target in target_set])
        probs /= probs.sum()
        return probs, target_set

    ############################################################
    # Calculations
    ############################################################

    def info_params(self):
        """
        Shows the parameters of the model.
        """
        PA.info_params(self)
        Homophily.info_params(self)

    def info_computed(self):
        """
        Shows the computed properties of the graph.
        """
        PA.info_computed(self)
        Homophily.info_computed(self)

    def infer_homophily_values(self) -> Tuple[float, float]:
        """
        Infers the level of homophily using the analytical solution of the model.

        Returns
        -------
        tuple[float, float]
            homophily between majority nodes, and homophily between minority nodes

        Notes
        -----
        See derivations in [Karimi2018]_.
        """

        h_MM, h_mm = infer_homophily(self)

        return h_MM, h_mm

    def _makecopy(self):
        """
        Makes a copy of the current object.
        """
        return self.__class__(n=self.n,
                              k=self.k,
                              f_m=self.f_m,
                              h_MM=self.h_MM,
                              h_mm=self.h_mm,
                              seed=self.seed)

    @staticmethod
    def fit(g, n=None, k=None, seed=None):
        """
        It fits the PAH model to the given graph.

        Parameters
        ----------
        g: netin.UnDiGraph
            graph to fit the model to

        n: int
            number of nodes to override (e.g., to generate a smaller network)

        k: int
            minimum node degree to override (e.g., to generate a denser network ``k>1``)

        seed: object
            seed for random number generator

        Returns
        -------
        netin.PAH
            fitted model
        """
        n = n or g.number_of_nodes()
        k = k or g.calculate_minimum_degree()
        f_m = g.calculate_fraction_of_minority()
        h_MM, h_mm = infer_homophily(g)

        new_g = PAH(n=n,
                    k=k,
                    f_m=f_m,
                    h_MM=float(h_MM),
                    h_mm=float(h_mm),
                    seed=seed)
        new_g.generate()

        return new_g


def infer_homophily(g) -> Tuple[float, float]:
    f_m = g.calculate_fraction_of_minority()
    f_M = 1 - f_m

    e = g.calculate_edge_type_counts()
    e_MM = e['MM']
    e_mm = e['mm']
    M = e['MM'] + e['mm'] + e['Mm'] + e['mM']

    p_MM = e_MM / M
    p_mm = e_mm / M

    pl_M, pl_m = g.calculate_degree_powerlaw_exponents()
    b_M = -1 / (pl_M + 1)
    b_m = -1 / (pl_m + 1)

    # equations
    hmm, hMM, hmM, hMm = symbols('hmm hMM hmM hMm')
    eq1 = Eq((f_m * f_m * hmm * (1 - b_M)) / ((f_m * hmm * (1 - b_M)) + (f_M * hmM * (1 - b_m))), p_mm)
    eq2 = Eq(hmm + hmM, 1)

    eq3 = Eq((f_M * f_M * hMM * (1 - b_m)) / ((f_M * hMM * (1 - b_m)) + (f_m * hMm * (1 - b_M))), p_MM)
    eq4 = Eq(hMM + hMm, 1)

    solution = solve((eq1, eq2, eq3, eq4), (hmm, hmM, hMM, hMm))
    h_MM, h_mm = solution[hMM], solution[hmm]
    return h_MM, h_mm
