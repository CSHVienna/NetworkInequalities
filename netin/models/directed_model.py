from typing import Dict, Optional, Any

import powerlaw
import numpy as np

from netin.graphs.graph import Graph

from ..graphs.directed import DiGraph
from ..graphs.node_attributes import NodeAttributes
from .model import Model

class DirectedModel(Model):
    node_activity: NodeAttributes
    d: float
    plo_M: float
    plo_m: float

    def __init__(
            self, N: int, f: float,
            d: float, plo_M: float, plo_m: float,
            graph: Optional[DiGraph] = None,
            seed: int = 1):
        assert graph is None or isinstance(graph, DiGraph), "graph must be a DiGraph"

        self.d = d
        self.plo_M = plo_M
        self.plo_m = plo_m

        super().__init__(N, f, graph, seed)
        self._initialize_node_activity()

    def _initialize_graph(self):
        self.graph = DiGraph()

    def _initialize_node_activity(self):
        act_M = powerlaw.Power_Law(
            parameters=[self.plo_M],
            discrete=True)\
                .generate_random(self.get_n_majority())
        act_m = powerlaw.Power_Law(
            parameters=[self.plo_m],
            discrete=True)\
                .generate_random(self.get_n_minority())

        mask_M = self.get_majority_mask()
        mask_m = self.get_minority_mask()

        a_node_activity = np.zeros(self.N)
        a_node_activity[mask_M] = act_M
        a_node_activity[mask_m] = act_m

        if np.inf in a_node_activity:
            a_node_activity[a_node_activity == np.inf] = 0.0
            a_node_activity += 1
        a_node_activity /= a_node_activity.sum()

        self.node_activity = NodeAttributes.from_ndarray(a_node_activity, name="node_activity")

    def get_metadata(
            self, d_meta_data: Optional[Dict[str, Any]] = None)\
                -> Dict[str, Any]:
        d = super().get_metadata(d_meta_data)
        d[self.__class__.__name__] = {
            "d": self.d,
            "plo_M": self.plo_M,
            "plo_m": self.plo_m
        }
        return d

    def simulate(self) -> Graph:
        raise NotImplementedError
