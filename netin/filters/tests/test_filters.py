import numpy as np
import pytest

from ..active_nodes import ActiveNodes
from ..no_double_links import NoDoubleLinks
from ..no_self_links import NoSelfLinks
from ..triadic_closure import TriadicClosure
from ...graphs.node_attributes import NodeVector
from ...graphs.graph import Graph
from ...graphs.directed import DiGraph

class TestFilters(object):
    N = 1000

    @staticmethod
    def assert_one_zero_xor(mask: np.ndarray):
        mask_one = mask == 1.
        mask_zero = mask == 0.

        assert np.all(mask_one ^ mask_zero), "Mask contains values other than 0 or 1."

    def test_active_nodes(self):
        ug = Graph()
        with pytest.raises(AssertionError):
            _ = ActiveNodes(N=self.N, graph=ug)

        dg = DiGraph()
        an = ActiveNodes(N=self.N, graph=dg)
        m0 = an.get_target_mask(-1)
        TestFilters.assert_one_zero_xor(m0)
        assert isinstance(m0, NodeVector), "Active nodes mask is not an instance of `NodeVector`."
        assert np.all(m0.attr() == 0.), "Active nodes mask is not initialized correctly."

        dg.add_edge(0, 1)
        m1 = an.get_target_mask(-1)
        TestFilters.assert_one_zero_xor(m1)
        assert m1[0] == 1.,\
            ("Active nodes mask is not updated correctly. "
             f"Node `0` is not active but `{m1[0]}`.")
        assert m1[1] == 0.,\
            ("Active nodes mask is not updated correctly. "
             f"Node `1` is not zero but `{m1[1]}`.")

    def test_no_self_links(self):
        nsl = NoSelfLinks(N=self.N)

        for i in range(self.N):
            msk = nsl.get_target_mask(i)
            TestFilters.assert_one_zero_xor(msk)
            assert msk[i] == 0., "Self link detected as valid."
            assert int(np.sum(msk)) == self.N - 1, f"Incorrect number of active nodes (sum={np.sum(msk)})."

    def test_no_double_links(self):
        ug = Graph()
        ndl = NoDoubleLinks(N=self.N, graph=ug)

        with pytest.raises(KeyError):
            _ = ndl.get_target_mask(0)

        ug.add_node(0)
        m1 = ndl.get_target_mask(0)
        TestFilters.assert_one_zero_xor(m1)
        assert np.all(m1 == 1.), "No double links mask is not initialized correctly."

        ug.add_edge(0, 1)
        m2 = ndl.get_target_mask(0)
        TestFilters.assert_one_zero_xor(m2)
        assert m2[1] == 0., "Double link detected as valid."
        assert np.sum(m2) == self.N - 1, f"Incorrect number of double links (sum={np.sum(m2)})."

        dg = DiGraph()
        ndl = NoDoubleLinks(N=self.N, graph=dg)
        dg.add_edge(0, 1)
        m3 = ndl.get_target_mask(0)
        TestFilters.assert_one_zero_xor(m3)
        assert m3[1] == 0., "Double link detected as valid."

        m4 = ndl.get_target_mask(1)
        TestFilters.assert_one_zero_xor(m4)
        assert m4[0] == 1., "Directedness not considered."


    def test_triadic_closure(self):
        g = Graph()
        tc = TriadicClosure(graph=g)

        g.add_edge(0, 1)
        g.add_edge(1, 2)
        m0 = tc.get_target_mask(0)
        TestFilters.assert_one_zero_xor(m0)
        assert m0[2] == 1., "Triadic closure not detected for link `0-2`."

        m1 = tc.get_target_mask(2)
        TestFilters.assert_one_zero_xor(m1)
        assert m1[0] == 1., "Triadic closure not detected for link `2-0`."

        m2 = tc.get_target_mask(1)
        TestFilters.assert_one_zero_xor(m2)
        assert np.all(m2 == 0.),\
            (f"False triadic closure detected for source `1` "
             f"(false targets: {np.where(m2 == 1.)}).")
