import numpy as np

from netin.models import DHModel

class TestDPAHModel(object):
    @staticmethod
    def _create_model(
        N=1000, d=0.005, f_m=0.1,
        plo_M=2.0, plo_m=2.0,
        h_M=0.2, h_m=0.9, seed=1234) -> DHModel:
        return DHModel(
            N=N, d=d, f_m=f_m, plo_M=plo_M, plo_m=plo_m, h_M=h_M, h_m=h_m, seed=seed)

    def test_simulation(self):
        model = TestDPAHModel._create_model()
        model.simulate()
        graph = model.graph

        assert len(graph) == model.N
        assert graph.is_directed()
        n_edges = graph.number_of_edges()
        assert np.isclose(
            n_edges / (model.N * (model.N - 1)),
            model.d,
            atol=1e-5)

    def test_preload_graph(self):
        pass

    def test_no_invalid_links(self):
        model = TestDPAHModel._create_model()
        model.simulate()
        graph = model.graph
        for node in graph.nodes():
            assert not graph.has_edge(node, node)
