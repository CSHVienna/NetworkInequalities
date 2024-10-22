from typing import Optional, List

import numpy as np

from ..utils.validator import validate_float, validate_int
from ..utils.constants import CLASS_ATTRIBUTE, CLASS_LABELS, MINORITY_VALUE, MAJORITY_VALUE
from .categorical_node_vector import CategoricalNodeVector

class BinaryClassNodeVector(CategoricalNodeVector):
    """Represents a binary class assignment for nodes

    Parameters
    ----------
    N : int
        Number of nodes.
    class_labels : Optional[List[str]], optional
        The class labels for the values, must be of size two, by default CLASS_LABELS
    name : Optional[str], optional
        Name for the NodeVector, by default CLASS_ATTRIBUTE
    """
    def __init__(
        self, N: int,
        class_labels: Optional[List[str]] = CLASS_LABELS,
        name: Optional[str] = CLASS_ATTRIBUTE) -> None:
        super().__init__(
            N=N, n_values=2,
            class_labels=class_labels,
            name=name)

    @classmethod
    def from_ndarray(cls,
        values: np.ndarray,
        class_labels: Optional[List[str]] = CLASS_LABELS,
        name: Optional[str] = None,
        **kwargs)\
            -> 'BinaryClassNodeVector':
        """Creates a BinaryClassNodeVector from a numpy array.

        Parameters
        ----------
        values : np.ndarray
            The values to be set.
        class_labels : Optional[List[str]], optional
            The class labels for the values, must be of size two, by default CLASS_LABELS
        name : Optional[str], optional
            Name for the NodeVector, by default CLASS_ATTRIBUTE
        """
        assert np.all((values == MINORITY_VALUE) | (values == MAJORITY_VALUE)),\
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
            N: int,
            f_m: float,
            class_labels: Optional[List[str]] = None,
            rng: Optional[np.random.Generator] = None,
            name: Optional[str] = None)\
                -> 'BinaryClassNodeVector':

        """Creates a BinaryClassNodeVector with a given fraction of minority nodes.

        Parameters
        ----------
        N : int
            Number of nodes.
        f_m : float
            Fraction of minority nodes.
        class_labels : Optional[List[str]], optional
            The class labels for the values, must be of size two, by default CLASS_LABELS
        rng : Optional[np.random.Generator], optional
            Random number generator, by default None
        name : Optional[str], optional
            Name for the NodeVector, by default CLASS_ATTRIBUTE
        """
        validate_float(f_m, minimum=0., maximum=0.5)
        validate_int(N, minimum=1)
        rng = np.random.default_rng() if rng is None else rng
        return cls.from_ndarray(
            values=np.where(rng.random(N) < f_m, MINORITY_VALUE, MAJORITY_VALUE),
            class_labels=class_labels,
            name=name)

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
        """Returns the number of nodes in the majority class.

        Returns
        -------
        int
            Number of nodes in the majority class.
        """
        return len(self) - self.get_n_minority()
