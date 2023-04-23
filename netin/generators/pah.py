from typing import Union, Set, Tuple

import numpy as np
from sympy import symbols
from sympy import Eq
from sympy import solve

from netin.utils import constants as const
from netin.generators.h import Homophily
from .pa import PA


class PAH(PA, Homophily):
    """Creates a new PAH instance.

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
    The initialization is an undirected with n nodes, where f_m are the minority.
    Then, everytime a node is selected as source, it gets connected to k target nodes.
    Target nodes are selected via preferential attachment (in-degree) and homophily (h_**).
    This model is based on [1] known as the "Barabasi model with homophily" or "BA Homophily".

    References
    ----------
    [1] F. Karimi, M. Génois, C. Wagner, P. Singer, & M. Strohmaier, M "Homophily influences ranking of minorities in social networks", Scientific reports 8(1), 11077, 2018.
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
        Infers the level of homopolily using the analutical solution of the model [1].

        Returns
        -------
        tuple[float, float]
            homophily between majority nodes, and homophily between minority nodes

        References
        ----------
        [1] F. Karimi, M. Génois, C. Wagner, P. Singer, & M. Strohmaier, M "Homophily influences ranking of minorities in social networks", Scientific reports 8(1), 11077, 2018.

        """
        f_m = self.calculate_fraction_of_minority()
        f_M = 1 - f_m

        e = self.count_edges_types()
        e_MM = e['MM']
        e_mm = e['mm']
        M = e['MM'] + e['mm'] + e['Mm'] + e['mM']

        p_MM = e_MM / M
        p_mm = e_mm / M

        pl_M, pl_m = self.calculate_degree_powerlaw_exponents()
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
