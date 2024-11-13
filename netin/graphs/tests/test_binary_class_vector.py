import numpy as np
import pytest

from ..binary_class_node_vector import BinaryClassNodeVector

class TestBinaryClassNodeVector:
    def test_init(self):
        n = 10
        class_labels = [0, 1]
        node_values = BinaryClassNodeVector(n=n, class_labels=class_labels)
        assert np.all(node_values == 0), "Initial values must be 0."
        assert np.all(node_values.class_labels == class_labels),\
            "Class labels must be equal."
        assert node_values.n_values == 2

    def test_fail_init(self):
        with pytest.raises(AssertionError):
            BinaryClassNodeVector(n=10, class_labels=[0])
        with pytest.raises(AssertionError):
            BinaryClassNodeVector(n=10, class_labels=[0, 1, 2])

    def test_from_ndarray(self):
        n = 10
        values = np.random.randint(0, 2, n)
        node_values = BinaryClassNodeVector.from_ndarray(
            values)
        assert np.all(node_values == values), "Values must be equal."

        with pytest.raises(AssertionError):
            BinaryClassNodeVector.from_ndarray(
                values=np.arange(3))

    def test_from_fraction(self):
        n = 1000
        f_m = 0.3
        rng = np.random.default_rng(1)
        node_values = BinaryClassNodeVector.from_fraction(
            n=n, f_m=f_m, rng=rng)
        assert np.isclose(np.mean(node_values), f_m, atol=.05),\
            "Mean must be close to `f_m`."
        assert node_values.n_values == 2
        assert np.all(node_values.class_labels == [0, 1])
