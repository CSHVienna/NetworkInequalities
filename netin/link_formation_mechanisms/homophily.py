from typing import Union
import numpy as np

from .link_formation_mechanism import LinkFormationMechanism
from ..graphs.node_vector import NodeVector
from ..graphs.categorical_node_vector import CategoricalNodeVector
from ..utils.validator import validate_float

class Homophily(LinkFormationMechanism):
    """The Homophily link formation mechanism.
    The probabilities to connect to a target node are determined
    by the group membership of the source and target nodes.
    This implementation accounts for a single group membership per node
    with an arbitrary number of groups.

    Parameters
    ----------
    node_class_values: CategoricalNodeVector
        The class assignment for each node (dimensions `n_nodes`).
        This is also used to infer the number of classes.
    homophily : Union[float, np.ndarray]
        The homophily value(s).
        If a single value is provided, the in-group links have a probability of
        `homophily` and out-group links have a uniform
        probability of `1 - homophily / n_class_values - 1`.
        If a matrix is provided, the probabilities are determined by the matrix values.
        The matrix must be symmetric and have the shape of (`n_class_values`, `n_class_values`).
        Moreover, row values have to sum up to 1.
    """
    node_class_values: CategoricalNodeVector
    h: np.ndarray

    def __init__(
            self,
            node_class_values: CategoricalNodeVector,
            homophily: Union[float, np.ndarray],
            ) -> None:
        n_class_values = node_class_values.n_values

        _max = np.max(node_class_values)
        assert _max < n_class_values,\
        ("Classes must be numbered form 0 to "
            f"`n_class_values`. Highest class was {_max} "
            f"and `n_class_values` is {n_class_values}.")

        assert node_class_values.vals().ndim == 1,\
            ("Node class values must be a 1D array with dimensions (n_nodes,). "
             "Multiple dimensions are not (yet) supported by this class.")
        if isinstance(homophily, float):
            validate_float(homophily, minimum=0., maximum=1.)
            self.h = np.full(
                (n_class_values,  n_class_values),
                (1 - homophily) / (n_class_values - 1))
            np.fill_diagonal(self.h, homophily)
        else:
            assert homophily.shape == (n_class_values, n_class_values),\
                ("Homophily matrix must have symmetric "
                 "shape of (`n_class_values`, `n_class_values`)="
                 f"{(n_class_values, n_class_values)} "
                 f"but is {homophily.shape}")
            assert np.all(np.sum(homophily, axis=1) == 1.),\
                ("Row values of the homophily matrix must sum up to 1. "
                 f"Matrix is {homophily}")
            self.h = homophily
        self.node_class_values = node_class_values
        super().__init__(N=len(node_class_values))

    def _get_target_probabilities(self, source: int) -> NodeVector:
        a_node_class_values = self.node_class_values
        class_source = a_node_class_values[source]
        p_target = self.h[class_source, a_node_class_values]
        return NodeVector.from_ndarray(p_target)
