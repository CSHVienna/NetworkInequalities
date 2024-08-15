############################################
# System dependencies
############################################
import operator
from typing import Tuple
from typing import Union

from networkx import DiGraph

from netin.utils.constants import CLASS_ATTRIBUTE
from . import constants as const
from .sampling import Sampling


############################################
# Class
############################################
class DegreeRank(Sampling):
    """Sampling by degree rank

    Parameters
    ----------
    g: netin.Graph | netin.DiGraph
        global network

    pseeds: float
        fraction of seeds to sample

    max_tries: int
        maximum number of tries to sample a subgraph with enough classes and edges

    random_seed: object
        seed for random number generator

    kwargs: dict
        additional parameters for the sampling method

        order: str
            order of nodes by degree. Options: "asc" | "desc"

    """

    ######################################################
    # Constructor
    ######################################################
    # def __init__(self, g: netin.Graph, pseeds: float, random_seed: object = None, **kwargs):
    def __init__(
            self,
            graph: DiGraph, pseeds: float,
            class_attribute: str = CLASS_ATTRIBUTE,
            random_seed: object = None, **kwargs):
        super().__init__(
            graph=graph, pseeds=pseeds, max_tries=1,
            class_attribute=class_attribute, random_seed=random_seed, **kwargs)
        self.order = self.kwargs.get("order", const.DESC)

    @property
    def method_name(self) -> str:
        name = f"{const.DEGREE_RANK} ({const.DESC if self.is_descending() else const.ASC})"
        return name

    def sampling(self):
        super().sampling()

    def is_ascending(self) -> bool:
        return not self.is_descending()

    def is_descending(self) -> bool:
        return self.order == const.DESC

    def _sample(self) -> Tuple[list, Union[list, None]]:
        """
        Creates a subgraph from G based on degree rank
        """
        edges = None

        ### 1. pick random nodes
        nodes = sorted([(n, d) for n, d in self.g.degree() if d > 0],
                       key=operator.itemgetter(1),
                       reverse=self.is_descending())
        nodes = nodes[:self.nseeds]
        nodes, degree = zip(*nodes)
        num_classes = self._count_classes(nodes)

        if num_classes < const.MIN_CLASSES:
            raise ValueError(f"{num_classes} class(es). Not enough classes in the sample of {self.nseeds} nodes."
                             "Try increasing the number of seeds or sampling by DegreeGroupRank.")

        return nodes, edges
