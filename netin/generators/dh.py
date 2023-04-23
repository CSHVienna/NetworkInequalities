from typing import Union, Set, Tuple

import numpy as np

from netin.generators.directed import DiGraph
from netin.generators.h import Homophily
from netin.utils import constants as const


class DH(DiGraph, Homophily):
    """ Creates a new DH instance. A directed graph with homophily.

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

    h_MM: float
        homophily within majority group (minimum=0, maximum=1)

    h_mm: float
        homophily within minority group (minimum=0, maximum=1)

    seed: object
        seed for random number generator

    Notes
    -----
    The initialization is a directed with n nodes and no edges.
    Source nodes are selected based on their activity given by plo_M (if majority) or plo_m (if minority).
    Target nodes are selected via homophily, see [Espin-Noboa2022]_.
    """

    ############################################################
    # Constructor
    ############################################################

    def __init__(self, n: int, d: float, f_m: float, plo_M: float, plo_m: float, h_MM: float, h_mm: float,
                 seed: object = None):
        DiGraph.__init__(self, n=n, d=d, f_m=f_m, plo_M=plo_M, plo_m=plo_m, seed=seed)
        Homophily.__init__(self, n=n, f_m=f_m, h_MM=h_MM, h_mm=h_mm, seed=seed)

    ############################################################
    # Init
    ############################################################

    def _infer_model_name(self):
        """
        Infers the name of the model.
        """
        return self.set_model_name(const.DH_MODEL_NAME)

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
        DiGraph._initialize(self, class_attribute, class_values, class_labels)
        Homophily._initialize(self, class_attribute, class_values, class_labels)

    def get_target_probabilities(self, source: Union[None, int], target_set: Union[None, Set[int], np.array],
                                 special_targets: Union[None, object, iter] = None) -> np.array:
        """
        Returns the probabilities of the target nodes to be selected given a source node.

        Parameters
        ----------
        source: int
            source node (id)

        target_set: set
            set of target nodes (ids)

        special_targets: object
            special targets

        Returns
        -------
        probs: np.array
            probabilities of the target nodes to be selected

        """
        probs, ts = Homophily.get_target_probabilities(self, source, target_set, special_targets)
        return probs

    ############################################################
    # Calculations
    ############################################################

    def info_params(self):
        """
        Shows the parameters of the model.
        """
        DiGraph.info_params(self)
        Homophily.info_params(self)

    def info_computed(self):
        """
        Shows the computed properties of the graph.
        """
        Homophily.info_computed(self)

    def infer_homophily_values(self) -> Tuple[float, float]:
        """
        Infers analytically the homophily values for the majority and minority classes.

        Returns
        -------
        h_MM: float
            homophily within majority group

        h_mm: float
            homophily within minority group
        """
        from sympy import symbols
        from sympy import Eq
        from sympy import solve

        f_m = self.calculate_fraction_of_minority()
        f_M = 1 - f_m

        e = self.count_edges_types()
        e_MM = e['MM']
        e_mm = e['mm']
        e_Mm = e['Mm']
        e_mM = e['mM']

        p_MM = e_MM / (e_MM + e_Mm)
        p_mm = e_mm / (e_mm + e_mM)

        # equations
        hmm, hMM, hmM, hMm = symbols('hmm hMM hmM hMm')
        eq1 = Eq((f_m * hmm) / ((f_m * hmm) + (f_M * hmM)), p_mm)
        eq2 = Eq(hmm + hmM, 1)

        eq3 = Eq((f_M * hMM) / ((f_M * hMM) + (f_m * hMm)), p_MM)
        eq4 = Eq(hMM + hMm, 1)

        solution = solve((eq1, eq2, eq3, eq4), (hmm, hmM, hMM, hMm))
        h_MM, h_mm = solution[hMM], solution[hmm]
        return h_MM, h_mm
