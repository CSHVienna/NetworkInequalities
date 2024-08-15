from typing import List

import pytest
import numpy as np

from ..homophily import Homophily
from ..two_class_homophily import TwoClassHomophily
from ...graphs.node_class_vector import NodeClassVector

class TestHomophily:
    @staticmethod
    def _generate_class_values(l_n: List[int]) -> NodeClassVector:
        l_classes = []
        for i, Ni in enumerate(l_n):
            for _ in range(Ni):
                l_classes.append(i)
        return NodeClassVector\
            .from_ndarray(np.array(l_classes))

    def test_homophily_mixing_matrix_dims(self):
        l_n = (3, 2, 1)
        classes = TestHomophily._generate_class_values(l_n)
        h_manual = Homophily(
            node_class_values=classes,
            homophily=0.5,
            n_class_values=3)
        assert h_manual.h.shape == (3, 3)

        h_auto = Homophily(
            node_class_values=classes,
            homophily=0.5)
        assert h_auto.h.shape == (3, 3)

        h_high = Homophily(
            node_class_values=classes,
            homophily=0.5,
            n_class_values=5)
        assert h_high.h.shape == (5, 5)

        with pytest.raises(AssertionError):
            _ = Homophily(
                node_class_values=classes,
                homophily=0.5,
                n_class_values=2)

    def test_homophily_value(self):
        l_n = (3, 2, 1)
        classes = TestHomophily._generate_class_values(l_n)
        h = Homophily(
            node_class_values=classes,
            homophily=0.5)
        assert np.all(h.h == np.array([
            [0.5, 0.25, 0.25],
            [0.25, 0.5, 0.25],
            [0.25, 0.25, 0.5]]))

        uniform = np.ones((3,3)) / 3
        h = Homophily(
            node_class_values=classes,
            homophily=uniform)
        assert np.all(h.h == uniform)

        invalid_low = np.asarray(
            [[0.5, 0.2, 0.2],
             [0.25, 0.5, 0.25],
             [0.25, 0.25, 0.5]])
        with pytest.raises(AssertionError):
            _ = Homophily(
                node_class_values=classes,
                homophily=invalid_low)

    def test_two_class_homophily(self):
        classes = TestHomophily._generate_class_values([3,2,1])
        uniform = np.ones((3,3)) / 3
        with pytest.raises(AssertionError):
            _ = TwoClassHomophily(
                node_class_values=classes,
                homophily=uniform)

        classes = TestHomophily._generate_class_values([3,2])
        h = TwoClassHomophily(
            node_class_values=classes,
            homophily=0.2)
        assert np.all(h.h == np.array([
            [0.2, 0.8],
            [0.8, 0.2]])), "Homophily matrix wrong."

        h = TwoClassHomophily.from_two_class_homophily(
            node_class_values=classes,
            homophily=(0.2, 0.6))
        assert np.all(h.h == np.array([
            [0.2, 0.8],
            [0.4, 0.6]])), "Homophily matrix wrong."

