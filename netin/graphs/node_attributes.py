from typing import Dict, Type, Optional, Any
from numbers import Number

import numpy as np

from ..base_class import BaseClass

class NodeAttributes(BaseClass):
    _attributes: np.ndarray
    name: str

    def __init__(self, N: int, dtype: Type, fill_value: Optional[Number] = None, name: Optional[str] = None) -> None:
        super().__init__()
        self._attributes = np.zeros(
            N,
            dtype=dtype) if fill_value is None else\
                np.full(N, fill_value, dtype=dtype)
        self.name = name

    def set_attributes(self, attributes: np.ndarray) -> None:
        """Sets the node attributes.

        Parameters
        ----------
        attributes : np.ndarray
            Node attributes.
        """
        self._attributes = attributes

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
        node_attributes = cls(attributes.shape[0], attributes.dtype, **kwargs)
        node_attributes.set_attributes(attributes)
        return node_attributes

    def __getitem__(self, key: int) -> np.ndarray:
        return self._attributes[key]

    def __setitem__(self, key: int, value: np.ndarray) -> None:
        self._attributes[key] = value

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
