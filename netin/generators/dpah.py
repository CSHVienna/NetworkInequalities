from typing import Union

import numpy as np

from netin.generators.h import Homophily
from netin.utils import constants as const
from .dpa import DPA


class DPAH(DPA, Homophily):
    """Creates a new DPAH instance. A directed graph with preferential attachment and homophily.

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
    The initialization is a directed graph with n nodes where f_m are the minority.
    Source nodes are selected based on their activity given by plo_M (if majority) or plo_m (if minority).
    Target nodes are selected via preferential attachment (in-degree) an homophily (h**).
    This model is based on [Espin-Noboa2022]_ which is the directed version of the "BA Homophily" model [Karimi2018]_.
    """

    ############################################################
    # Constructor
    ############################################################

    def __init__(self, n: int, d: float, f_m: float, plo_M: float, plo_m: float, h_MM: float, h_mm: float,
                 seed: object = None):
        DPA.__init__(self, n=n, d=d, f_m=f_m, plo_M=plo_M, plo_m=plo_m, seed=seed)
        Homophily.__init__(self, n=n, f_m=f_m, h_MM=h_MM, h_mm=h_mm, seed=seed)
        self.model_name = const.DPAH_MODEL_NAME

    ############################################################
    # Generation
    ############################################################

    def initialize(self, class_attribute: str = 'm', class_values: list = None, class_labels: list = None):
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
        DPA.initialize(self, class_attribute, class_values, class_labels)
        Homophily.initialize(self, class_attribute, class_values, class_labels)

    def get_target_probabilities(self, source: int, available_nodes: Union[None, list[int]]) -> np.array:
        """
        Returns the probabilities of selecting a target node from a set of nodes based on
        preferential attachment and homophily, i.e., in-degree or target and homophily between source and target.

        Parameters
        ----------
        source: int
            source node

        available_nodes: Set[int]
            set of target nodes

        Returns
        -------
        np.array
            probabilities of selecting a target node from a set of nodes
        """
        probs = np.array([self.get_homophily_between_source_and_target(source, target) *
                          (self.get_in_degree(target) + const.EPSILON) for target in available_nodes])
        probs /= probs.sum()
        return probs

    ############################################################
    # Calculations
    ############################################################

    def info_params(self):
        """
        Shows the (input) parameters of the model.
        """
        DPA.info_params(self)
        Homophily.info_params(self)

    def info_computed(self):
        """
        Shows the (computed) properties of the graph.
        """
        DPA.info_computed(self)
        Homophily.info_computed(self)

    def infer_homophily_values(self) -> tuple[float, float]:
        """
        Infers the level of homophily within the majority and minority groups analytically.

        Returns
        -------
        Tuple[float, float]
            homophily within the majority and minority groups
        """
        h_MM, h_mm = None, None
        return h_MM, h_mm

    def makecopy(self):
        """
        Makes a copy of the current object.
        """
        obj = self.__class__(n=self.n,
                             d=self.d,
                             f_m=self.f_m,
                             plo_M=self.plo_M,
                             plo_m=self.plo_m,
                             h_MM=self.h_MM,
                             h_mm=self.h_mm,
                             seed=self.seed)

        # @TODO: check if this is necessary
        # obj.initialize(class_attribute=self.class_attribute,
        #                 class_values=self.class_values,
        #                 class_labels=self.class_labels)
        return obj
