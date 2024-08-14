from typing import Optional

import numpy as np

from ..utils.validator import validate_float, validate_int
from .node_vector import NodeVector

class BinaryMinorityNodeVector(NodeVector):
    @classmethod
    def from_ndarray(cls, values: np.ndarray, **kwargs) -> 'BinaryMinorityNodeVector':
        assert values.dtype == np.int, "values must be of type np.int"
        assert np.all(np.isin(values, {0, 1})), "values must be binary"
        return super().from_ndarray(values, **kwargs)

    @classmethod
    def from_fraction(
            cls,
            N: int, fraction: float,
            rng: Optional[np.random.Generator] = None) -> 'BinaryMinorityNodeVector':
        validate_float(fraction, minimum=0., maximum=0.5)
        validate_int(N, minimum=1)
        rng = np.random.default_rng() if rng is None else rng
        return cls.from_ndarray(np.where(rng.rand(N) < fraction, 1, 0))

    def get_minority_mask(self) -> np.ndarray:
        """Returns the mask of the minority class.

        Returns
        -------
        np.ndarray
            Mask of the minority class.
        """
        return self == 1

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
