from typing import Dict, Optional, Any, Union
from abc import ABC, abstractmethod

import numpy as np

from ..graphs.graph import Graph
from ..base_class import BaseClass
from ..filters.no_double_links import NoDoubleLinks
from ..filters.no_self_links import NoSelfLinks
from ..utils.event_handling import HasEvents, Event

class Model(ABC, HasEvents, BaseClass):
    """Model class.
    Abstract class that defines a growing network model.
    Specific growing-network-model implementations should inherit
    from this class and implement the provided abstract methods.
    """
    SHORT = "MODEL"

    N: int
    graph: Graph
    seed: int

    _n_nodes_total: int

    _f_no_double_links: NoDoubleLinks
    _f_no_self_links: NoSelfLinks

    _rng: np.random.Generator

    EVENTS = [Event.SIMULATION_START, Event.SIMULATION_END]

    def __init__(
            self, *args,
            N: int,
            seed: Union[int, np.random.Generator] = 1,
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
        if isinstance(seed, int):
            self._rng = np.random.default_rng(seed=seed)
        elif isinstance(seed, np.random.Generator):
            self._rng = seed
        else:
            raise ValueError(f"`seed` must be an `int` or `np.random.Generator` but is {type(seed)}")

    def initialize_simulation(self):
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
        raise NotImplementedError

    @abstractmethod
    def _initialize_lfms(self):
        raise NotImplementedError

    @abstractmethod
    def _initialize_node_classes(self):
        raise NotImplementedError

    @abstractmethod
    def _initialize_empty_graph(self) -> Graph:
        raise NotImplementedError

    @abstractmethod
    def _populate_initial_graph(self) -> Graph:
        raise NotImplementedError

    def _initialize_graph(self):
        self.graph = self._initialize_empty_graph()
        self._populate_initial_graph()
        self._n_nodes_total = self.N

    def _initialize_filters(self):
        self._f_no_self_links = NoSelfLinks(
            N=self._n_nodes_total)
        self._f_no_double_links = NoDoubleLinks(
            N=self._n_nodes_total,
            graph=self.graph)

    def simulate(self) -> Graph:
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
        return self._f_no_self_links.get_target_mask(source)\
            * self._f_no_double_links.get_target_mask(source)

    def get_metadata(self, d_meta_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        d = super().get_metadata(d_meta_data)
        d[self.__class__.__name__] = {
            "N": self.N,
            "seed": self.seed
        }
        self.graph.get_metadata(
            d[self.__class__.__name__])

        return d

    def _sample_target_node(
            self, target_probabilities: np.ndarray) -> int:
        return self._rng.choice(
            len(target_probabilities),
            p=target_probabilities)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(N={self.N})"
