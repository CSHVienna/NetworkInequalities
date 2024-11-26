import pytest

from netin.models import PAHModel
from netin.utils.constants import CLASS_ATTRIBUTE
from netin.graphs.binary_class_node_vector import BinaryClassNodeVector
from netin.graphs import Graph, DiGraph

import numpy as np

class TestPAHModel:
    @staticmethod
    def _create_model(
        n=1000, k=2, f_m=0.1, h_mm=0.5, h_MM=0.5, seed=1234
    ):
        return PAHModel(
            n=n, k=k, f_m=f_m, h_mm=h_mm, h_MM=h_MM, seed=seed
        )

    def test_simulation(self):
        n = 1000
        k = 2
        f_m = .3

        model = TestPAHModel._create_model(n=n, k=k, f_m=f_m)
        model.simulate()

        assert model.graph is not None
        assert not model.graph.is_directed()
        assert len(model.graph) == n
        _sum_links = sum(model.graph.degree(v)\
                         for v in model.graph.nodes())
        assert (_sum_links // 2) == (((n - k) * k) + (k * (k - 1)) // 2)

        node_classes = model.graph.get_node_class(CLASS_ATTRIBUTE)
        assert node_classes is not None
        assert len(node_classes) == n
        assert np.isclose(np.mean(node_classes), f_m, atol=0.05)

        assert isinstance(node_classes, BinaryClassNodeVector)
        assert np.isclose(
            node_classes.get_n_minority(),
            f_m * n, rtol=0.05)

    def test_preload_graph(self):
        n = 1000
        n_pre = 10
        k = 2
        f_m = .3
        f_m_pre = .1

        model = TestPAHModel._create_model(n=n, f_m=f_m)
        with pytest.raises(AssertionError):
            model.preload_graph(DiGraph())

        g_pre = Graph()
        for i in range(n_pre):
            g_pre.add_node(i)
        g_pre.add_edge(0, 1)
        g_pre.set_node_class(
            CLASS_ATTRIBUTE,
            BinaryClassNodeVector.from_fraction(n=n_pre, f_m=f_m_pre))
        model.preload_graph(g_pre)
        model.initialize_simulation()
        assert len(g_pre.get_node_class(CLASS_ATTRIBUTE)) == (n_pre + n),\
            "Node class not set correctly."
        model.simulate()

        assert len(model.graph) == (n + n_pre)
        _sum_links = sum(model.graph.degree(v)\
                         for v in model.graph.nodes()) // 2
        assert _sum_links == (1 + (n * k))
        for u in range(n_pre):
            for v in range(u):
                if u == v:
                    assert not model.graph.has_edge(u, v)
                elif (u == 0 and v == 1) or (v == 0 and u == 1):
                    assert model.graph.has_edge(u, v)
                else:
                    assert not model.graph.has_edge(u, v)
        node_classes = model.graph.get_node_class(CLASS_ATTRIBUTE)
        assert node_classes is not None
        assert len(node_classes) == (n + n_pre)
        assert np.isclose(np.mean(node_classes), f_m, atol=0.05)

        model = TestPAHModel._create_model(n=n, f_m=f_m)
        g_pre = Graph()
        for i in range(n_pre):
            g_pre.add_node(i)
        g_pre.set_node_class(
            CLASS_ATTRIBUTE,
            BinaryClassNodeVector.from_fraction(n=n_pre, f_m=f_m_pre))
        model.simulate()
        node_classes = model.graph.get_node_class(CLASS_ATTRIBUTE)
        assert node_classes is not None
        assert len(node_classes) == n
        assert np.isclose(
            np.sum(node_classes),
            (n_pre * f_m_pre) + (n * f_m),
            rtol=0.05)

    def test_no_invalid_links(self):
        n = 1000
        k = 2
        model = TestPAHModel._create_model(n=n, k=k)
        model.simulate()
        graph = model.graph

        # The graph class cannot store double links
        # Hence, if there are no self-links, the number of links
        # must be `(n-k) * k` because each but the first `k` nodes
        # create `k` links.
        for u in graph.nodes():
            assert not graph.has_edge(u, u)
        n_links = graph.number_of_edges()
        assert n_links == ((n - k) * k) + ((k * (k - 1)) // 2)

    def test_heterophily_min_advantage(self):
        h = 0.1

        model = TestPAHModel._create_model(
            h_mm=h, h_MM=h)
        model.simulate()
        node_classes = model.graph.get_node_class(CLASS_ATTRIBUTE)

        minority_mask = node_classes.get_minority_mask()
        majority_mask = node_classes.get_majority_mask()

        degrees = model.graph.degrees()
        degrees_min = degrees[minority_mask]
        degrees_maj = degrees[majority_mask]

        assert np.mean(degrees_min) > np.mean(degrees_maj)

    def test_homophily_maj_advantage(self):
        h = .9

        model = TestPAHModel._create_model(
            h_mm=h, h_MM=h)
        model.simulate()
        node_classes = model.graph.get_node_class(CLASS_ATTRIBUTE)

        minority_mask = node_classes.get_minority_mask()
        majority_mask = node_classes.get_majority_mask()

        degrees = model.graph.degrees()
        degrees_min = degrees[minority_mask]
        degrees_maj = degrees[majority_mask]

        assert np.mean(degrees_min) < np.mean(degrees_maj)

    def test_seeding(self):
        model1 = TestPAHModel._create_model()
        model1.simulate()
        graph1 = model1.graph
        nc1 = graph1.get_node_class(CLASS_ATTRIBUTE)

        model2 = TestPAHModel._create_model()
        model2.simulate()
        graph2 = model2.graph
        nc2 = graph2.get_node_class(CLASS_ATTRIBUTE)

        model3 = TestPAHModel._create_model(seed=999)
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
