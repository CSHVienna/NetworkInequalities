from abc import abstractmethod

import numpy as np

from ..graphs import Graph

class LinkFormationMechanism:
    graph: Graph

    def __init__(self, graph: Graph) -> None:
        self.graph = graph

    @abstractmethod
    def get_target_probabilities(self, source: int) -> np.ndarray:
        raise NotImplementedError
