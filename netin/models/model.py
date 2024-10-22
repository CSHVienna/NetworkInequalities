from typing import Dict, Optional, Any, Union
from abc import ABC, abstractmethod

import numpy as np

from ..graphs.graph import Graph
from ..base_class import BaseClass
from ..filters.no_double_links import NoDoubleLinks
from ..filters.no_self_links import NoSelfLinks
from ..utils.event_handling import HasEvents, Event

class Model(ABC, HasEvents, BaseClass):
    """Abstract modelling class.
    This class defines a growing network model.
    Specific growing-network-model implementations should inherit
    from this class and implement the provided abstract methods.
    """

    SHORT = "MODEL"
    """A shorthand for the model name"""

    N: int
    graph: Graph
    seed: int

    _n_nodes_total: int

    _f_no_double_links: NoDoubleLinks
    _f_no_self_links: NoSelfLinks

    _rng: np.random.Generator

    EVENTS = [Event.SIMULATION_START, Event.SIMULATION_END]
    """Evokes :attr:`.Event.SIMULATION_START` and :attr:`.Event.SIMULATION_END`.

    :meta hide-value:"""

    def __init__(
            self, *args,
            N: int,
            seed: Optional[Union[int, np.random.Generator]] = None,
            **kwargs):
        """Creates a new instance of the Model class.

        Parameters
        ----------
        N : int
            Number of nodes to be added to the network.
        seed : int, optional
            A random seed for reproducibility, by default 1
        """
        super().__init__(*args, **kwargs)

        self.N = N
        self.graph = None

        self._set_seed(seed)

    def _set_seed(self, seed: Union[int, np.random.Generator]):
        """Sets the seed for the random number generator.

        Parameters
        ----------
        seed : int
            Seed for the random number generator.
        """
        if isinstance(seed, int) or (seed is None):
            self._rng = np.random.default_rng(seed=seed)
        elif isinstance(seed, np.random.Generator):
            self._rng = seed
        else:
            raise ValueError(f"`seed` must be an `int` or `np.random.Generator` but is {type(seed)}")

    def initialize_simulation(self):
        """Initializes the simulation of the model.
        The order of initialization is

        0. (Optional) Graph initialization if not preloaded.
        1. Node classes initialization.
        2. Filters initialization.
        3. Link formation mechanisms initialization.
        """

        self.log(f"Initializing simulation of {self.__class__.__name__}")
        if self.graph is None:
            self.log("Initializing graph")
            self._initialize_graph()
        else:
            self.log("Working with preloaded graph")
        self.log("Initializing node classes")
        self._initialize_node_classes()
        self.log("Initializing filters")
        self._initialize_filters()
        self.log("Initializing link formation mechanisms")
        self._initialize_lfms()

    @abstractmethod
    def _simulate(self) -> Graph:
        """Abstract simulation method.
        This method should contain the actual simulation logic of the model.
        Should be overwritten by the specific model implementation.

        Returns
        -------
        Graph
            The simulated graph.
        """
        raise NotImplementedError

    @abstractmethod
    def _initialize_lfms(self):
        """Initialize the link formation mechanisms.
        This should be overwritten by the actual model implementations.
        See :func:`._simulate` for the initialization order.
        """
        raise NotImplementedError

    @abstractmethod
    def _initialize_node_classes(self):
        """Initializes the node classes.
        This should be overwritten by the actual model implementations.
        See :func:`._simulate` for the initialization order.
        """
        raise NotImplementedError

    @abstractmethod
    def _initialize_empty_graph(self) -> Graph:
        """Initializes an empty graph.
        This should be overwritten by the actual model implementations.
        See :meth:`.Model._simulate` for the initialization order.
        """
        raise NotImplementedError

    @abstractmethod
    def _populate_initial_graph(self) -> Graph:
        """Populates the initial graph.
        This should be overwritten by the actual model implementations.
        See :meth:`.Model._simulate` for the initialization order.

        Returns
        -------
        Graph
            The populated graph.
        """
        raise NotImplementedError

    def _initialize_graph(self):
        """Initializes the empty graph, populates it and set the number of total nodes.
        """
        self.graph = self._initialize_empty_graph()
        self._populate_initial_graph()
        self._n_nodes_total = self.N

    def _initialize_filters(self):
        """Initializes the filters.
        Default filters are no self loops and no double links.
        """
        self._f_no_self_links = NoSelfLinks(
            N=self._n_nodes_total)
        self._f_no_double_links = NoDoubleLinks(
            N=self._n_nodes_total,
            graph=self.graph)

    def simulate(self) -> Graph:
        """Runs the simulation.
        This calls :meth:`.Model.initialize_simulation` and :meth:`.Model._simulate`.
        Check these functions in case you want to extend :class:`Model`.
        Triggers the :attr:`.Event.SIMULATION_START` and :attr:`.Event.SIMULATION_END` events.

        Notes
        -----
        Triggers the following events.

        - :attr:`.Event.SIMULATION_START`: When the simulation starts.
        - :attr:`.Event.SIMULATION_END`: When the simulation ends.

        Returns
        -------
        :class:`.Graph`
            The simulated graph.
        """
        self.log(f"Simulating {self.__class__.__name__}")
        self.trigger_event(event=Event.SIMULATION_START)
        self.initialize_simulation()
        graph = self._simulate()
        self.trigger_event(event=Event.SIMULATION_END)
        self.log(f"Done with simulation of {self.__class__.__name__}")
        return graph

    def preload_graph(self, graph: Graph):
        """Preloads a graph into the model.

        Parameters
        ----------
        graph : Graph
            Graph to preload.
        """
        assert len(graph) <= self.N,\
            "The graph has more nodes than the final number of nodes."
        self.graph = graph
        self._n_nodes_total = self.N + len(graph)

    def compute_target_probabilities(self, source: int) -> np.ndarray:
        """Computes the target probabilities.
        This function applies the default filters to avoid self loops and no double links.

        Parameters
        ----------
        source : int
            The source node.

        Returns
        -------
        numpy.ndarray
            Array of target probabilities.
        """
        return self._f_no_self_links.get_target_mask(source)\
            * self._f_no_double_links.get_target_mask(source)

    def get_metadata(self, d_meta_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Returns or updates the metadata.
        This sets the number of nodes and random seed.
        Model implementations should update this to add other model parameters.

        Parameters
        ----------
        d_meta_data : Optional[Dict[str, Any]], optional
            Dictionary containing object metadata, by default None

        Returns
        -------
        Dict[str, Any]
            Metadata dictionary.
        """
        d = super().get_metadata(d_meta_data)
        d[self.__class__.__name__] = {
            "N": self.N,
            "seed": self.seed
        }
        self.graph.get_metadata(
            d[self.__class__.__name__])

        return d

    def _add_edge_to_graph(self, source: int, target: int):
        """Adds an edge to the graph.
        This triggers the :attr:`.Event.LINK_ADD_BEFORE` and :attr:`.Event.LINK_ADD_AFTER` events.

        Parameters
        ----------
        source : int
            Source node.
        target : int
            Target node.
        """
        self.trigger_event(event=Event.LINK_ADD_BEFORE, source=source, target=target)
        self.graph.add_edge(source, target)
        self.trigger_event(event=Event.LINK_ADD_AFTER, source=source, target=target)

    def _sample_target_node(
            self, target_probabilities: np.ndarray)\
                -> int:
        """Picks a target node.

        Returns
        -------
        int
            Target node.
        """
        return self._rng.choice(
            len(target_probabilities),
            p=target_probabilities)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(N={self.N})"
