from typing import Union
import numpy as np

from .link_formation_mechanism import LinkFormationMechanism

class Homophily(LinkFormationMechanism):
    node_class_values: np.ndarray
    h: np.ndarray

    def __init__(
            self,
            node_class_values: np.ndarray,
            n_class_values: int,
            homophily: Union[float, np.ndarray]) -> None:
        """Initializes the Homophily link formation mechanism.
        The probabilities to connect to a target node are determined
        by the group membership of the source and target nodes.
        Currently, this class accounts for a single group membership per node
        with an arbitrary number of classes.

        Parameters
        ----------
        node_class_values : np.ndarray
            The class assignment for each node (dimensions `n_nodes`).
            Classes must be integers in the range of [0, `n_class_values`).
        n_class_values : int
            Number of classes.
        homophily : Union[float, np.ndarray]
            The homophily value(s).
            If a single value is provided, the in-group links have a probability of
            `homophily` and out-group links have a probability of `1 - homophily`.
            If a matrix is provided, the probabilities are determined by the matrix values.
        """
        super().__init__()

        assert node_class_values.ndim == 1,\
            ("Node class values must be a 1D array with dimensions (n_nodes,). "
             "Multiple dimensions are not (yet) supported by this class.")
        if isinstance(homophily, float):
            self.h = np.full(n_class_values * n_class_values, homophily)
        else:
            assert homophily.shape == (n_class_values, n_class_values),\
                "Homophily matrix must have the same shape as the node class values matrix"
            self.h = homophily
        self.node_class_values = node_class_values

    def get_target_probabilities(self, source: int) -> np.ndarray:
        class_source = self.node_class_values[source]
        p_target = self.h[class_source, self.node_class_values]
        return p_target / p_target.sum()
