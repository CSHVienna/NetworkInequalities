from typing import Dict, Type, Optional, Any
from numbers import Number

import numpy as np

from ..base_class import BaseClass
from ..utils.validator import validate_int

class NodeAttributes(BaseClass):
    _attributes: np.ndarray
    __array_interface__: Any
    name: str

    def __init__(self, N: int, dtype: Type, fill_value: Optional[Number] = None, name: Optional[str] = None) -> None:
        validate_int(N, minimum=1)
        super().__init__()
        self.set_attributes(np.zeros(
            N,
            dtype=dtype) if fill_value is None else\
                np.full(N, fill_value, dtype=dtype))
        self.name = name

    def set_attributes(self, attributes: np.ndarray) -> None:
        """Sets the node attributes.

        Parameters
        ----------
        attributes : np.ndarray
            Node attributes.
        """
        self._attributes = attributes
        self.__array_interface__ = attributes.__array_interface__

    def get_attributes(self) -> Any:
        return self._attributes

    def attr(self) -> np.ndarray:
        return self.get_attributes()

    @classmethod
    def from_ndarray(
            cls,
            attributes: np.ndarray,
            **kwargs) -> "NodeAttributes":
        """Creates a new instance of the NodeAttributes class from a numpy array.

        Parameters
        ----------
        attributes : np.ndarray
            The values of the node attributes.
        """
        assert isinstance(attributes, np.ndarray),\
            f"attributes must be of type np.ndarray, but is {type(attributes)}"
        node_attributes = cls(len(attributes), attributes.dtype, **kwargs)
        node_attributes.set_attributes(attributes)
        return node_attributes

    def get_metadata(self, d_meta_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        d = super().get_metadata(d_meta_data)
        name_cls = self.__class__.__name__
        d_meta_data = {
            "name": self.name,
            "attributes": self._attributes.tolist()
        }
        if name_cls in d:
            if self.name in d[name_cls]:
                d[name_cls][self.name].append(d_meta_data)
            else:
                d[name_cls][self.name] = [d_meta_data]
        else:
            d[name_cls] = {self.name: [d_meta_data]}

        return d

    def sum(self, *args, **kwargs) -> Any:
        return self._attributes.sum(*args, **kwargs)

    def __eq__(self, value: object):
        return self._attributes.__eq__(value)

    def __getitem__(self, key: int) -> np.ndarray:
        return self._attributes[key]

    def __setitem__(self, key: int, value: np.ndarray) -> None:
        self._attributes[key] = value

    def __mul__(self, other):
        return np.multiply(self, other)

    def __rmul__(self, other):
        return np.multiply(other, self)

    def __array__(self, dtype=None, copy=None):
        # This allows the usage of numpy functions
        # Such as np.sum(node_attributes)
        return self._attributes.__array__(dtype=dtype, copy=copy)
