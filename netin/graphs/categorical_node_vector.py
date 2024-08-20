from numbers import Number
from typing import Optional, List

import numpy as np

from .node_vector import NodeVector

class CategoricalNodeVector(NodeVector):
    n_values: int
    class_labels: np.ndarray

    def __init__(
            self, N: int,
            n_values: int,
            node_labels: Optional[List[str]] = None,
            class_labels: Optional[List[str]] = None,
            fill_value: Optional[Number] = None,
            name: Optional[str] = None) -> None:
        assert n_values is None or n_values > 0, "n_values must be positive"
        assert class_labels is None or len(class_labels) == n_values,\
            "class_labels must have the same length as n_values"

        self.n_values = n_values
        self.class_labels = np.asarray(class_labels)\
            if class_labels is not None else np.arange(n_values)

        super().__init__(
            N=N, dtype=int,
            fill_value=fill_value, name=name)

    @classmethod
    def from_ndarray(
        cls,
        values: np.ndarray,
        n_values: Optional[int] = None,
        name: Optional[str] = None,
        class_labels: Optional[List[str]] = None,
        **kwargs)\
            -> 'CategoricalNodeVector':
        assert values.dtype == int, "values must be of type int"
        assert 0 <= np.min(values), "values must be non-negative"
        assert n_values is None or n_values > np.max(values),\
            "n_values must be greater or equal to the maximum value in values"

        ncv = CategoricalNodeVector(
            N=len(values),
            n_values=n_values\
                if n_values is not None else np.max(values) + 1,
            name=name,
            class_labels=class_labels, **kwargs)
        ncv._values = values

        return ncv

    def get_class_values(self) -> np.ndarray:
        return self.class_labels[self._values]
