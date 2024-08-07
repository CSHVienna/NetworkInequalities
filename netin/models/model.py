from typing import Dict, Optional, Any
from abc import ABC, abstractmethod

import numpy as np

from ..graphs.graph import Graph
from ..graphs.node_attributes import NodeAttributes
from ..base_class import BaseClass

class Model(ABC, BaseClass):
    """Model class.
    Abstract class that defines a growing network model.
    Specific growing-network-model implementations should inherit from this class and implement the provided abstract methods.
    """
    N: int
    f: float
    graph: Graph
    node_minority_class: NodeAttributes
    seed: int

    def __init__(
            self, N: int, f: float,
            graph: Optional[Graph] = None,
            seed: int = 1):
        """Creates a new instance of the Model class.

        Parameters
        ----------
        N : int
            Number of final nodes in the network.
        f : float
            Fraction of nodes that belong to the minority class.
        graph : Optional[Graph], optional
            If present, an existing network that will be extended. In this case, `N >= graph.number_of_nodes()` as the graph will be extended by the missing nodes. If no graph is given, the model creates its own graph and initializes it with `m` fully connected nodes.
            Calling the `simulate`-function will then add the remaining `N - m` nodes.
        seed : int, optional
            A random seed for reproducibility, by default 1
        """
        super().__init__()

        self.N = N
        self.f = f
        self.seed = seed
        np.random.seed(seed)

        self.node_minority_class = NodeAttributes\
            .from_ndarray(
                np.where(np.random.rand(N) < f, 1, 0), name="minority_class")

        if graph is None:
            self._initialize_graph()
            self._populate_initial_graph()
        self._initialize_lfms()

    def _initialize_graph(self):
        """Initializes an empty graph.
        Function can be overwritten by subclasses.
        """
        self.graph = Graph()

    @abstractmethod
    def _populate_initial_graph(self):
        raise NotImplementedError

    @abstractmethod
    def _initialize_lfms(self):
        pass

    @abstractmethod
    def compute_target_probabilities(self, source: int) -> np.ndarray:
        pass

    @abstractmethod
    def simulate(self) -> Graph:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(N={self.N}, f={self.f})"

    def get_minority_mask(self) -> np.ndarray:
        """Returns the mask of the minority class.

        Returns
        -------
        np.ndarray
            Mask of the minority class.
        """
        return self.node_minority_class == 1

    def get_majority_mask(self) -> np.ndarray:
        """Returns the mask of the majority class.

        Returns
        -------
        np.ndarray
            Mask of the majority class.
        """
        return ~self.get_minority_mask()

    def get_n_minority(self) -> int:
        """Returns the number of nodes in the minority class.

        Returns
        -------
        int
            Number of nodes in the minority class.
        """
        return np.sum(self.node_minority_class)

    def get_n_majority(self) -> int:
        return self.N - self.get_n_minority()

    @staticmethod
    def _sample_target_node(target_probabilities: np.ndarray) -> int:
        return np.random.choice(
            np.arange(len(target_probabilities)),
            p=target_probabilities)

    def get_metadata(self, d_meta_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        d = super().get_metadata(d_meta_data)
        d[self.__class__.__name__] = {
            "N": self.N,
            "f": self.f,
            "seed": self.seed
        }
        self.graph.get_metadata(
            d[self.__class__.__name__])
        self.node_minority_class.get_metadata(
            d[self.__class__.__name__])

        return d
