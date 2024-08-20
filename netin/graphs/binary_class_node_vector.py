from typing import Optional, List

import numpy as np

from ..utils.validator import validate_float, validate_int
from ..utils.constants import CLASS_ATTRIBUTE, CLASS_LABELS, MINORITY_VALUE, MAJORITY_VALUE
from .categorical_node_vector import CategoricalNodeVector

class BinaryClassNodeVector(CategoricalNodeVector):
    def __init__(
        self, N: int,
        node_labels: Optional[List[str]] = None,
        class_labels: Optional[List[str]] = CLASS_LABELS,
        name: Optional[str] = CLASS_ATTRIBUTE) -> None:
        super().__init__(
            N=N, n_values=2,
            class_labels=class_labels,
            node_labels=node_labels,
            name=name)

    @classmethod
    def from_nd_array(cls,
        values: np.ndarray,
        class_labels: Optional[List[str]] = None,
        name: Optional[str] = None,
        **kwargs)\
            -> 'BinaryClassNodeVector':
        assert np.all(np.isin(values, {MAJORITY_VALUE, MINORITY_VALUE})),\
            "values must be binary"
        bmnv = BinaryClassNodeVector(
            N=len(values),
            name=name,
            class_labels=class_labels, **kwargs)
        bmnv._values = values
        return bmnv

    @classmethod
    def from_fraction(
            cls,
            N: int, f_m: float,
            node_labels: Optional[List[str]] = None,
            class_labels: Optional[List[str]] = None,
            rng: Optional[np.random.Generator] = None)\
                -> 'BinaryClassNodeVector':
        validate_float(f_m, minimum=0., maximum=0.5)
        validate_int(N, minimum=1)
        rng = np.random.default_rng() if rng is None else rng
        return cls.from_ndarray(
            values=np.where(rng.random(N) < f_m, MINORITY_VALUE, MAJORITY_VALUE),
            node_labels=node_labels,
            class_labels=class_labels)

    def get_minority_mask(self) -> np.ndarray:
        """Returns the mask of the minority class.

        Returns
        -------
        np.ndarray
            Mask of the minority class.
        """
        return self == MINORITY_VALUE

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
        return np.sum(self)

    def get_n_majority(self) -> int:
        return len(self) - self.get_n_minority()
