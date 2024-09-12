import pytest

from netin.models import PAModel
from netin.utils.constants import CLASS_ATTRIBUTE
from netin.graphs.binary_class_node_vector import BinaryClassNodeVector
from netin.graphs import Graph, DiGraph

import numpy as np

class TestPAModel:
    @staticmethod
    def _create_model(
        N=1000, m=2, f_m=0.1, seed=1234
    ):
        return PAModel(
            N=N, m=m, f_m=f_m, seed=seed
        )

    def test_simulation(self):
        N = 1000
        m = 2
        f_m = .3

        model = TestPAModel._create_model(N=N, m=m, f_m=f_m)
        model.simulate()

        assert model.graph is not None
        assert not model.graph.is_directed()
        assert len(model.graph) == N
        _sum_links = sum(model.graph.degree(v)\
                         for v in model.graph.nodes())
        assert (_sum_links // 2) == (((N - m) * m) + (m * (m - 1)) // 2)

        node_classes = model.graph.get_node_class(CLASS_ATTRIBUTE)
        assert node_classes is not None
        assert len(node_classes) == N
        assert np.isclose(np.mean(node_classes), f_m, atol=0.05)

        assert isinstance(node_classes, BinaryClassNodeVector)
        assert np.isclose(
            node_classes.get_n_minority(),
            f_m * N, rtol=0.05)

    def test_preload_graph(self):
        N = 1000
        N_pre = 10
        m = 2
        f_m = .3
        f_m_pre = .1

        model = TestPAModel._create_model(N=N, f_m=f_m)
        with pytest.raises(AssertionError):
            model.preload_graph(DiGraph())

        g_pre = Graph()
        for i in range(N_pre):
            g_pre.add_node(i)
        g_pre.add_edge(0, 1)
        g_pre.set_node_class(
            CLASS_ATTRIBUTE,
            BinaryClassNodeVector.from_fraction(N=N_pre, f_m=f_m_pre))
        model.preload_graph(g_pre)
        model.initialize_simulation()
        assert len(g_pre.get_node_class(CLASS_ATTRIBUTE)) == (N_pre + N),\
            "Node class not set correctly."
        model.simulate()

        assert len(model.graph) == (N + N_pre)
        _sum_links = sum(model.graph.degree(v)\
                         for v in model.graph.nodes()) // 2
        assert _sum_links == (1 + (N * m))
        for u in range(N_pre):
            for v in range(u):
                if u == v:
                    assert not model.graph.has_edge(u, v)
                elif (u == 0 and v == 1) or (v == 0 and u == 1):
                    assert model.graph.has_edge(u, v)
                else:
                    assert not model.graph.has_edge(u, v)
        node_classes = model.graph.get_node_class(CLASS_ATTRIBUTE)
        assert node_classes is not None
        assert len(node_classes) == (N + N_pre)
        assert np.isclose(np.mean(node_classes), f_m, atol=0.05)

        model = TestPAModel._create_model(N=N, f_m=f_m)
        g_pre = Graph()
        for i in range(N_pre):
            g_pre.add_node(i)
        g_pre.set_node_class(
            CLASS_ATTRIBUTE,
            BinaryClassNodeVector.from_fraction(N=N_pre, f_m=f_m_pre))
        model.simulate()
        node_classes = model.graph.get_node_class(CLASS_ATTRIBUTE)
        assert node_classes is not None
        assert len(node_classes) == N
        assert np.isclose(
            np.sum(node_classes),
            (N_pre * f_m_pre) + (N * f_m),
            rtol=0.05)

    def test_no_invalid_links(self):
        N = 1000
        m = 2
        model = TestPAModel._create_model(N=N, m=m)
        model.simulate()
        graph = model.graph

        # The graph class cannot store double links
        # Hence, if there are no self-links, the number of links
        # must be `(N-m) * m` because each but the first `m` nodes
        # create `m` links.
        for u in graph.nodes():
            assert not graph.has_edge(u, u)
        n_links = graph.number_of_edges()
        assert n_links == ((N - m) * m) + ((m * (m - 1)) // 2)

    def test_seeding(self):
        model1 = TestPAModel._create_model(seed=1234)
        model1.simulate()
        graph1 = model1.graph
        nc1 = graph1.get_node_class(CLASS_ATTRIBUTE)

        model2 = TestPAModel._create_model(seed=1234)
        model2.simulate()
        graph2 = model2.graph
        nc2 = graph2.get_node_class(CLASS_ATTRIBUTE)

        model3 = TestPAModel._create_model(seed=999)
        model3.simulate()
        graph3 = model3.graph
        nc3 = graph3.get_node_class(CLASS_ATTRIBUTE)

        assert np.all(nc1 == nc2)
        assert not np.all(nc1 == nc3)

        _n_non_overlap = 0
        for u in graph1.nodes():
            assert graph1[u] == graph2[u]
            if not graph1[u] == graph3[u]:
                _n_non_overlap += 1
        assert _n_non_overlap > 0