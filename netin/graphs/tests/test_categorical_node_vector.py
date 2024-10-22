import numpy as np
import pytest

from ..categorical_node_vector import CategoricalNodeVector

class TestCategoricalNodeVector:
    def test_init(self):
        N = 10
        n_values = 3
        class_labels = ["a", "b", "c"]
        node_values = CategoricalNodeVector(
            N=N, n_values=n_values, class_labels=class_labels)
        assert np.all(node_values == 0), "Initial values must be 0."
        assert np.all(node_values.class_labels == class_labels),\
            "Class labels must be equal."

    def test_fail_init(self):
        with pytest.raises(AssertionError):
            CategoricalNodeVector(N=10, n_values=0)
        with pytest.raises(AssertionError):
            CategoricalNodeVector(N=10, n_values=3, class_labels=["a", "b"])

    def test_from_ndarray(self):
        N = 10
        values = np.random.randint(0, 3, N)
        class_labels = ["a", "b", "c"]
        node_values = CategoricalNodeVector.from_ndarray(
            values, n_values=3, class_labels=class_labels)
        assert np.all(node_values == values), "Values must be equal."
        assert np.all(node_values.class_labels == class_labels),\
            "Class labels must be equal."

        with pytest.raises(AssertionError):
            CategoricalNodeVector.from_ndarray(
                values=np.arange(4), n_values=3)

    def test_get_class_values(self):
        N = 10
        values = np.random.randint(0, 3, N)
        class_labels = ["a", "b", "c"]
        node_values = CategoricalNodeVector.from_ndarray(
            values, n_values=3, class_labels=class_labels)
        class_values = node_values.get_class_values()
        assert np.all(class_values == np.array(class_labels)[values]),\
            "Class values must be equal."
