import numpy as np
import pytest

from ..node_vector import NodeVector

class TestNodeVector:
    def test_np_functions(self):
        N = 10
        values = np.arange(N)
        node_values = NodeVector.from_ndarray(values)

        _sum = np.sum(node_values)

        assert _sum == N * (N - 1) // 2, "Sum must be `N * N(N-1) // 2` for np.arange(N)."
        assert isinstance(_sum, np.integer), "Sum must be an integer."
        assert np.max(node_values) == N - 1, "Max value must be N - 1."
        assert np.min(node_values) == 0, "Min value must be 0."

        mask = node_values > 5
        assert np.sum(mask) == 4, "There must be 4 values greater than 5."
        assert np.all(node_values[mask] == np.arange(6, 10)), "Values must be 6, 7, 8, 9."

    def test_custom_node_labels(self):
        nodes = "abcde"
        values = np.arange(len(nodes))
        nv = NodeVector.from_ndarray(values, node_labels=list(nodes))

        assert nv["a"] == 0, "Value for node `a` must be 0."
        assert nv["e"] == 4, "Value for node `e` must be 4."
        assert np.all(nv["a":"d"] == np.arange(0, 3)),\
            "Values for nodes `a` to `d` must be 0 to 3."
        with pytest.raises(KeyError):
            _ = nv[0]
        assert np.all(nv[np.asarray(["a", "c", "e"])] == np.array([0, 2, 4])),\
            "Values for nodes np.array([`a`, `c`, `e`]) must be 0, 2, 4."
        assert np.all(nv[["a", "c", "e"]] == np.array([0, 2, 4])),\
            "Values for nodes `a`, `c`, `e` must be 0, 2, 4."

