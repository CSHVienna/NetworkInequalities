from numbers import Number
from typing import Optional, List

import numpy as np

from .node_vector import NodeVector

class NodeClassVector(NodeVector):
    n_values: int
    class_labels: Optional[List[str]] = None

    def __init__(
            self, N: int,
            n_values: int,
            class_labels: Optional[List[str]] = None,
            fill_value: Optional[Number] = None,
            name: Optional[str] = None) -> None:
        assert n_values is None or n_values > 0, "n_values must be positive"
        assert class_labels is None or len(class_labels) == n_values,\
            "class_labels must have the same length as n_values"

        self.n_values = n_values
        self.class_labels = class_labels

        super().__init__(N, int, fill_value, name)

    @classmethod
    def from_ndarray(
        cls,
        values: np.ndarray,
        n_values: Optional[int] = None,
        class_labels: Optional[List[str]] = None,
        **kwargs)\
            -> 'NodeClassVector':
        assert values.dtype == np.int, "values must be of type np.int"
        assert 0 <= np.min(values), "values must be non-negative"
        assert n_values is None or n_values > np.max(values),\
            "n_values must be greater or equal to the maximum value in values"

        ncv = NodeClassVector(
            N=len(values),
            n_values=n_values\
                if n_values is not None else np.max(values) + 1,
            class_labels=class_labels, **kwargs)

        return ncv

    def get_class_values(self, as_labels: bool=False) -> np.ndarray:
        return np.arange(self.n_values) if not as_labels else self.class_labels
