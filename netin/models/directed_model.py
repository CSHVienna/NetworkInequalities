from typing import Dict, Optional, Any, Union

import powerlaw
import numpy as np
import networkx as nx

from netin.graphs.graph import Graph

from .binary_class_model import BinaryClassModel
from ..graphs.directed import DiGraph
from ..graphs.node_vector import NodeVector
from ..utils.event_handling import Event
from ..utils import constants as const
from ..utils.validator import validate_float, validate_int
from ..utils.constants import CLASS_ATTRIBUTE
from ..filters.active_nodes import ActiveNodes
from ..link_formation_mechanisms.uniform import Uniform

class DirectedModel(BinaryClassModel):
    """The DirectedModel defines a network growth model which selects source nodes
    based on their activity and parameterized by power law exponents ``plo_m/M``.
    Edges are created until a total network density ``d`` is reached.
    Subclasses should implement the way in which target nodes are chosen.
    See [Espin-Noboa2022]_ for details.

    Parameters
    ----------
    N : int
        The number of nodes to be added.
    f_m : float
        The fraction of minority nodes.
    d : float
        Edge density to be reached.
    plo_M : float
        Power law exponent for the majority activity.
    plo_m : float
        Power law exponent for the minority activity.
    seed : Union[int, np.random.Generator], optional
        Randomization seed or random number generator, by default 1
    """

    SHORT = "DIRECTED"
    d: float
    plo_M: float
    plo_m: float

    _node_activity: NodeVector
    _f_active_nodes: ActiveNodes
    _lfm_uniform: Uniform
    _out_degrees: NodeVector

    def __init__(
            self, *args,
            N: int, f_m: float,
            d: float, plo_M: float, plo_m: float,
            seed: Optional[Union[int, np.random.Generator]] = None,
            **kwargs):
        validate_float(d, minimum=0., maximum=1.)
        validate_float(plo_M, minimum=0.)
        validate_float(plo_m, minimum=0.)
        validate_float(f_m, minimum=0., maximum=1.)
        validate_int(N, minimum=1)

        super().__init__(
            *args, N=N, seed=seed, f_m=f_m,
            **kwargs)

        self.d = d
        self.plo_M = plo_M
        self.plo_m = plo_m

    def _populate_initial_graph(self):
        # Add nodes without links
        for i in range(self.N):
            self.graph.add_node(i)
        return self.graph

    def _initialize_lfms(self):
        # Initialize out-degrees for activity-based node selection
        self._out_degrees = NodeVector(
            self._n_nodes_total,
            dtype=int, name="out_degrees")
        self.graph.register_event_handler(
            event=Event.LINK_ADD_AFTER,
            function=self._update_out_degrees)

        self._lfm_uniform = Uniform(
            self._n_nodes_total)

        self._initialize_node_activity()

    def _initialize_filters(self):
        super()._initialize_filters()
        # Add filter for active nodes
        self._f_active_nodes = ActiveNodes(
            N=self._n_nodes_total,
            graph=self.graph)

    def _initialize_node_activity(self):
        # Generate node activity based on power law distributions
        minority_class = self.graph.get_node_class(CLASS_ATTRIBUTE)
        act_M = powerlaw.Power_Law(
            parameters=[self.plo_M],
            discrete=True)\
                .generate_random(minority_class.get_n_majority())
        act_m = powerlaw.Power_Law(
            parameters=[self.plo_m],
            discrete=True)\
                .generate_random(minority_class.get_n_minority())

        mask_M = minority_class.get_majority_mask()
        mask_m = minority_class.get_minority_mask()

        a_node_activity = np.zeros(self._n_nodes_total)
        a_node_activity[mask_M] = act_M
        a_node_activity[mask_m] = act_m

        if np.inf in a_node_activity:
            a_node_activity[a_node_activity == np.inf] = 0.0
            a_node_activity += 1
        a_node_activity /= a_node_activity.sum()

        self._node_activity = NodeVector.from_ndarray(
            a_node_activity, name="node_activity")

    def _initialize_empty_graph(self) -> Graph:
        return DiGraph()

    def _update_out_degrees(self, source: int, _: int):
        self._out_degrees[source] += 1

    def _get_expected_number_of_edges(self):
        # Compute the expected number of edges to reach the desired density
        return int(
            round(
                self.d\
                * self._n_nodes_total\
                * (self._n_nodes_total - 1)))

    def _get_sources(self):
        # Choose source nodes based on their activity
        return np.random.choice(
            a=np.arange(self._n_nodes_total),
            size=self._get_expected_number_of_edges(),
            replace=True,
            p=self._node_activity)

    def _get_target(self, source: int):
        # Initialize uniform probabilities
        target_probabilities = self._lfm_uniform.get_target_probabilities(source)

        # Check if there are enough edges to consider only nodes with out_degree > 0
        one_percent = self._n_nodes_total * 1 / 100.
        if np.count_nonzero(self._out_degrees) > one_percent:
            # if there are enough edges, then select only nodes
            # with out_degree > 0 that are not already
            # connected to the source.
            # Having out_degree > 0 means they are nodes
            # that have been in the network for at least one time step
            target_probabilities *= self._f_active_nodes.get_target_mask(source)

        # Call other potential link formation mechanisms
        target_probabilities *= self.compute_target_probabilities(source)

        # Check if all probabilities are zero
        # https://stackoverflow.com/questions/18395725/test-if-numpy-array-contains-only-zeros
        if not np.any(target_probabilities):
            return None

        # Normalize probabilities
        target_probabilities /= target_probabilities.sum()

        return self._sample_target_node(
            target_probabilities=target_probabilities
        )

    def get_metadata(
            self, d_meta_data: Optional[Dict[str, Any]] = None)\
                -> Dict[str, Any]:
        """Returns the metadata of the model.

        Returns
        -------
        Dict[str, Any]
            Metadata of the model.
            It includes the density ``d``, the activity power law exponents ``plo_M`` and ``plo_m``.
        """
        d = super().get_metadata(d_meta_data)
        d[self.__class__.__name__] = {
            "d": self.d,
            "plo_M": self.plo_M,
            "plo_m": self.plo_m
        }
        self._out_degrees.get_metadata(d[self.__class__.__name__])
        return d

    def preload_graph(self, _: DiGraph):
        """Preloads a graph into the model.
        This is currently not supported for the directed models.

        Parameters
        ----------
        _ : DiGraph
            The graph to preload.
        """
        raise NotImplementedError("Preloading is not supported for this model")

    def _simulate(self) -> DiGraph:
        tries = 0
        # Run until desired density is reached
        while self.graph.number_of_edges()\
                < self._get_expected_number_of_edges():
            tries += 1

            # Iterate through sources (based on activity)
            for source in self._get_sources():
                target = self._get_target(source)

                if target is None:
                    continue

                self._add_edge_to_graph(source, target)

                if self.graph.number_of_edges() >= self._get_expected_number_of_edges():
                    break

            # if no more edges can be added, break
            if tries > const.MAX_TRIES_EDGE\
                and (self.graph.number_of_edges() <\
                     self._get_expected_number_of_edges()):
                print((
                    f">> Edge density ({nx.density(self.graph)}) "
                    f"might differ from {self.d:.5f} ({self})"))
                break
        return self.graph
