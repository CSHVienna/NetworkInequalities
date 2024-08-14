from typing import Dict, Type, Optional, Any
from numbers import Number

import numpy as np

from ..base_class import BaseClass
from ..utils.validator import validate_int

class NodeVector(BaseClass):
    _values: np.ndarray
    __array_interface__: Any
    name: str

    def __init__(self, N: int, dtype: Type,
                 fill_value: Optional[Number] = None,
                 name: Optional[str] = None) -> None:
        validate_int(N, minimum=1)
        super().__init__()
        self.set_values(np.zeros(
            N,
            dtype=dtype) if fill_value is None else\
                np.full(N, fill_value, dtype=dtype))
        self.name = name

    def set_values(self, values: np.ndarray) -> None:
        """Sets the node values.

        Parameters
        ----------
        values : np.ndarray
            Node values.
        """
        self._values = values
        self.__array_interface__ = values.__array_interface__

    def get_values(self) -> Any:
        return self._values

    def vals(self) -> np.ndarray:
        return self.get_values()

    @classmethod
    def from_ndarray(
            cls,
            values: np.ndarray,
            **kwargs) -> "NodeVector":
        """Creates a new instance of the NodeVector class from a numpy array.

        Parameters
        ----------
        values : np.ndarray
            The values of the node values.
        """
        assert isinstance(values, np.ndarray),\
            f"values must be of type np.ndarray, but is {type(values)}"
        node_values = cls(len(values), values.dtype, **kwargs)
        node_values.set_values(values)
        return node_values

    def get_metadata(self, d_meta_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        d = super().get_metadata(d_meta_data)
        name_cls = self.__class__.__name__
        d_meta_data = {
            "name": self.name,
            "values": self._values.tolist()
        }
        if name_cls in d:
            if self.name in d[name_cls]:
                d[name_cls][self.name].append(d_meta_data)
            else:
                d[name_cls][self.name] = [d_meta_data]
        else:
            d[name_cls] = {self.name: [d_meta_data]}

        return d

    def __len__(self) -> int:
        return len(self._values)

    def __geq__(self, value: object) -> np.ndarray:
        return self._values.__geq__(value)

    def __leq__(self, value: object) -> np.ndarray:
        return self._values.__leq__(value)

    def __gt__(self, value: object) -> np.ndarray:
        return self._values.__gt__(value)

    def __lt__(self, value: object) -> np.ndarray:
        return self._values.__lt__(value)

    def sum(self, *args, **kwargs) -> Any:
        return self._values.sum(*args, **kwargs)

    def __eq__(self, value: object):
        return self._values.__eq__(value)

    def __getitem__(self, key: int) -> np.ndarray:
        return self._values[key]

    def __setitem__(self, key: int, value: np.ndarray) -> None:
        self._values[key] = value

    def __add__(self, other):
        return np.add(self, other)

    def __radd__(self, other):
        return np.add(other, self)

    def __mul__(self, other):
        return np.multiply(self, other)

    def __rmul__(self, other):
        return np.multiply(other, self)

    def __array__(self, dtype=None, copy=None):
        # This allows the usage of numpy functions
        # Such as np.sum(node_values)
        return self._values.__array__(dtype=dtype, copy=copy)
