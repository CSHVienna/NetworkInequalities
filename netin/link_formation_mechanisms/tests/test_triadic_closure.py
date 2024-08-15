import numpy as np

from ...graphs.graph import Graph
from ..triadic_closure import TriadicClosure

class TestTriadicClosure:
    def test_triadic_closure(self):
        g = Graph()
        tc = TriadicClosure(graph=g)

        g.add_edge(0, 1)
        g.add_edge(1, 2)
        m0 = tc.get_target_probabilities(0)
        assert m0[2] == 1, "Triadic closure not detected for link `0-2`."

        m1 = tc.get_target_probabilities(2)
        assert m1[0] == 1., "Triadic closure not detected for link `2-0`."

        m2 = tc.get_target_probabilities(1)
        assert np.all(m2 == m2[0]),\
            (f"False triadic closure detected for source `1` "
             f"(false targets: {np.where(m2 == 1.)}).")
