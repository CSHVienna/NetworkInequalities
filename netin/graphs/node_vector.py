from typing import Dict, Type, Optional, Any, Hashable, List
from numbers import Number

import numpy as np

from ..base_class import BaseClass
from ..utils.validator import validate_int

class NodeVector(BaseClass):
    _values: np.ndarray
    _map_node_labels: Dict[Hashable, int]
    name: str

    def __init__(self, N: int, dtype: Type,
                 node_labels: Optional[List[Hashable]] = None,
                 fill_value: Optional[Number] = None,
                 name: Optional[str] = None) -> None:
        validate_int(N, minimum=1)
        node_labels = np.arange(N)\
            if node_labels is None else node_labels
        assert len(node_labels) == N,\
            "`node_values` must have the same length as `N`"
        self._map_node_labels = {
            node_label: i for i, node_label in enumerate(node_labels)}
        self.set_values(np.zeros(
            N,
            dtype=dtype) if fill_value is None else\
                np.full(N, fill_value, dtype=dtype))
        self.name = name
        super().__init__()

    @classmethod
    def from_ndarray(
            cls,
            values: np.ndarray,
            node_labels: Optional[List[Hashable]] = None,
            **kwargs) -> "NodeVector":
        """Creates a new instance of the NodeVector class from a numpy array.

        Parameters
        ----------
        values : np.ndarray
            The values of the node values.
        """
        assert isinstance(values, np.ndarray),\
            f"values must be of type np.ndarray, but is {type(values)}"
        node_values = cls(
            N=len(values), dtype=values.dtype,
            node_labels=node_labels, **kwargs)
        node_values.set_values(values)
        return node_values

    def to_dict(self)\
        -> Dict[Hashable, int]:
        return {
            node_label: self[node_label]\
                for node_label in self._map_node_labels.keys()
        }

    def set_values(self, values: np.ndarray) -> None:
        """Sets the node values.

        Parameters
        ----------
        values : np.ndarray
            Node values.
        """
        self._values = values

    def get_values(self) -> Any:
        return self._values

    def get_labels(self) -> List[Hashable]:
        return list(self._map_node_labels.keys())

    def vals(self) -> np.ndarray:
        return self.get_values()

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

    def __getitem__(self, key) -> np.ndarray:
        if isinstance(key, slice):
            # Convert custom slice indices to numeric indices
            start = self._map_node_labels.get(key.start, 0) if key.start else None
            stop = self._map_node_labels.get(key.stop, len(self)) if key.stop else None
            step = key.step  # No need to convert step if it exists

            numeric_slice = slice(start, stop, step)
            return self._values[numeric_slice]
        if isinstance(key, list):
            mapped_indices = [
                self._map_node_labels[idx] for idx in key]
            return self._values[mapped_indices]
        if isinstance(key, np.ndarray):
            if key.dtype == bool:
                # Boolean indexing
                return self._values[key]
            mapped_indices = [
                self._map_node_labels[idx] for idx in key]
            return self._values[mapped_indices]
        return self._values[self._map_node_labels[key]]

    def __setitem__(self, key: Hashable, value: np.ndarray) -> None:
        self._values[self._map_node_labels[key]] = value

    def __add__(self, other):
        return np.add(self, other)

    def __radd__(self, other):
        return np.add(other, self)

    def __mul__(self, other):
        return np.multiply(self, other)

    def __rmul__(self, other):
        return np.multiply(other, self)

    def __array__(self, dtype=None):
        # This allows the usage of numpy functions
        # Such as np.sum(node_values)
        if dtype is not None:
            return self._values.astype(dtype)
        return self._values
