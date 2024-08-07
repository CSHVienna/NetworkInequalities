from typing import Dict, Optional, Any

import powerlaw
import numpy as np
import networkx as nx

from netin.graphs.graph import Graph

from ..graphs.directed import DiGraph
from ..graphs.node_attributes import NodeAttributes
from ..graphs.event import Event
from .model import Model
from ..utils import constants as const

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
        self._in_degrees = NodeAttributes(N, dtype=int, name="in_degrees")
        self._out_degrees = NodeAttributes(N, dtype=int, name="out_degrees")

        super().__init__(N, f, graph, seed)
        self._initialize_node_activity()

    def _initialize_graph(self):
        self.graph = DiGraph()
        self.graph.register_event_handler(
            event=Event.LINK_ADD_AFTER, function=self._update_node_degrees
        )

    def _populate_initial_graph(self):
        self.graph.add_nodes_from(range(self.N))

    def _get_expected_number_of_edges(self):
        return int(round(self.d * self.N * (self.N - 1)))

    def _get_sources(self):
        return np.random.choice(
            a=np.arange(self.N),
            size=self._get_expected_number_of_edges(),
            replace=True,
            p=self.node_activity)

    def _get_target(self, source: int):
        a_out_degrees = self._out_degrees.attr()

        # Initialize uniform probabilities
        # @TODO: Model these as link formation mechanisms
        target_probabilities = np.full(self.N, 1/self.N)
        target_probabilities[source] = 0. # Exclude self selection
        target_probabilities[self.graph[source]] = 0. # Exclude existing links
        one_percent = self.N * 1 / 100.
        if np.count_nonzero(a_out_degrees) > one_percent:
            # if there are enough edges, then select only nodes with out_degree > 0 that are not already
            # connected to the source.
            # Having out_degree > 0 means they are nodes that have been in the network for at least one time step
            target_probabilities *= np.where(a_out_degrees > 0, 1., 0.)

        # Check if all probabilities are zero
        # Follows https://stackoverflow.com/questions/18395725/test-if-numpy-array-contains-only-zeros
        if not np.any(target_probabilities):
            return None

        # Add potential link formation mechanisms
        target_probabilities = self.compute_target_probabilities(source)

        return self._sample_target_node(
            target_probabilities=target_probabilities
        )

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

    def _update_node_degrees(self, source: int, target: int):
        self._in_degrees[target] += 1
        self._out_degrees[source] += 1

    def get_metadata(
            self, d_meta_data: Optional[Dict[str, Any]] = None)\
                -> Dict[str, Any]:
        d = super().get_metadata(d_meta_data)
        d[self.__class__.__name__] = {
            "d": self.d,
            "plo_M": self.plo_M,
            "plo_m": self.plo_m
        }
        self._in_degrees.get_metadata(d[self.__class__.__name__])
        self._out_degrees.get_metadata(d[self.__class__.__name__])
        return d

    def simulate(self) -> Graph:
        tries = 0
        while self.graph.number_of_edges() < self._get_expected_number_of_edges():
            tries += 1
            for source in self._get_sources():
                target = self._get_target(source)

                if target is None:
                    continue

                if self.graph.number_of_edges() >= self._get_expected_number_of_edges():
                    break

            # if no more edges can be added, break
            if tries > const.MAX_TRIES_EDGE\
                and (self.graph.number_of_edges() <\
                     self._get_expected_number_of_edges()):
                print((
                    f">> Edge density ({nx.density(self)}) "
                    f"might differ from {self.d:.5f} (N={self.N}, f={self.f}"
                    f"seed={self.seed}, plo_M={self.plo_M}, plo_m={self.plo_m}"))
                break
