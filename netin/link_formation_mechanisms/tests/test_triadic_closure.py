import pytest
import numpy as np

from ...graphs.graph import Graph
from ..triadic_closure import TriadicClosure

class TestTriadicClosure:
    def test_triadic_closure(self):
        n = 3
        g = Graph()

        tc = TriadicClosure(n=n, graph=g)

        for n in range(n):
            g.add_node(n)

        g.add_edge(0, 1)
        g.add_edge(1, 2)

        m0 = tc.get_target_probabilities(0)
        assert m0[2] == 1., "Triadic closure not detected for link `0-2`."
        assert m0[0] == 0., "False triadic closure detected for link `0-0`."

        m1 = tc.get_target_probabilities(2)
        assert m1[0] == 1., "Triadic closure not detected for link `2-0`."
        assert m1[2] == 0., "False triadic closure detected for link `0-0`."

        m2 = tc.get_target_probabilities(1)
        assert np.all(m2 == m2[0]),\
            (f"False triadic closure detected for source `1` "
             f"(false targets: {np.where(m2 != 0.)}).")

    def test_varying_n(self):
        n = 5
        g = Graph()
        tc = TriadicClosure(n=n, graph=g)

        for node in range(n - 2):
            g.add_node(node)
        g.add_edge(0, 1)

        m = tc.get_target_probabilities(0)
        assert len(m) == n, "Incorrect number of targets."

    def test_initialization(self):
        n = 5
        g = Graph()

        for node in range(n):
            g.add_node(node)
        g.add_edge(0, 1)
        g.add_edge(1, 2)

        tc = TriadicClosure(n=n, graph=g)

        m = tc.get_target_probabilities(0)
        assert m[2] >= 0., "Triadic closure not detected for link `0-2`."
        assert m[0] >= 0., "Triadic closure not detected for link `0-2`."

    def test_repeated_source(self):
        n = 5
        g = Graph()
        tc = TriadicClosure(n=n, graph=g)

        for node in range(n):
            g.add_node(node)

        g.add_edge(1, 2)
        g.add_edge(3, 4)

        g.add_edge(0, 1)
        m1 = tc.get_target_probabilities(0)
        assert m1[2] == 1., "Triadic closure not detected for link `0-2`."
        assert m1[0] == 0., "False triadic closure detected for self-loop."

        g.add_edge(0, 3)
        m2 = tc.get_target_probabilities(0)
        assert m2[0] == 0., "False triadic closure detected for self-loop."
        assert m2[2] == .5, "Triadic closure not detected for link `0-2`."
        assert m2[4] == .5, "Triadic closure not detected for link `0-4`."

    def test_no_tc(self):
        pass

    def test_invalid_g(self):
        n = 5
        n_g = 10

        g = Graph()
        for node in range(n_g):
            g.add_node(node)

        with pytest.raises(AssertionError):
            _ = TriadicClosure(n=n, graph=g)
