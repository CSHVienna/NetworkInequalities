############################################
# System dependencies
############################################
import operator
from typing import Tuple
from typing import Union

import netin
from . import constants as const
from .degree_rank import DegreeRank


############################################
# Class
############################################
class DegreeGroupRank(DegreeRank):
    """Sampling by degree group rank (by degree rank per class)

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
    def __init__(self, g: netin.Graph, pseeds: float, random_seed: object = None, **kwargs):
        super().__init__(g=g, pseeds=pseeds, random_seed=random_seed, **kwargs)
        self.order = self.kwargs.get("order", const.DESC)

    @property
    def method_name(self) -> str:
        name = f"{const.DEGREE_GROUP_RANK} ({const.DESC if self.is_descending() else const.ASC})"
        return name

    def sampling(self):
        super().sampling()

    def _sample(self) -> Tuple[list, Union[list, None]]:
        """
        Creates a subgraph from G based on degree rank
        """
        nodes = []
        edges = None

        ### 1. pick nodes
        _nodes = {}
        for class_value in self.g.get_class_values():
            valid = [(n, d) for n, d in self.g.degree() if d > 0 and
                     self.g.nodes[n][self.g.get_class_attribute()] == class_value]
            _nodes[class_value] = sorted(valid,
                                         key=operator.itemgetter(1),
                                         reverse=self.is_descending())
            _nodes[class_value], _ = zip(*_nodes[class_value])

        while len(nodes) < self.nseeds:
            for class_value in self.g.get_class_values():
                _nodes[class_value] = list(_nodes[class_value])
                if len(_nodes[class_value]) > 0:
                    nodes.append(_nodes[class_value].pop(0))

        num_classes = self._count_classes(nodes)

        if num_classes < const.MIN_CLASSES:
            raise ValueError(f"{num_classes} class(es). Not enough classes in the sample of {self.nseeds} nodes."
                             "Try increasing the number of seeds or sampling by DegreeGroupRank.")

        return nodes, edges
