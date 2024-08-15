import numpy as np

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
