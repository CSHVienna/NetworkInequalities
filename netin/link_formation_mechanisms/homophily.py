import numpy as np

from netin.graphs import Graph
from .link_formation_mechanism import LinkFormationMechanism

class Homophily(LinkFormationMechanism):
    node_class_values: np.ndarray
    h: float

    def __init__(self, node_class_values: np.ndarray, h: float) -> None:
        super().__init__()
        self.h=h
        self.node_class_values = node_class_values

    def get_target_probabilities(self, source: int) -> np.ndarray:
        class_source = self.node_class_values[source]
        p_target = np.where(class_source == self.node_class_values, self.h, 1 - self.h)
        return p_target / p_target.sum()
