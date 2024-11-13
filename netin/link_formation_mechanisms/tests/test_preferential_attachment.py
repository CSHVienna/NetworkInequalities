import pytest

from ..preferential_attachment import PreferentialAttachment
from ...graphs.graph import Graph
from ...graphs.directed import DiGraph

class TestPreferentialAttachment:
    @staticmethod
    def create_graph(n=5, links=((0,1), (0,2), (0,3), (1,2))):
        g = Graph()

        for node in range(n):
            g.add_node(node)

        for source, target in links:
            g.add_edge(source, target)

        return g

    def test_pa(self):
        g = TestPreferentialAttachment.create_graph()
        pa = PreferentialAttachment(graph=g, n=len(g))

        tp = pa.get_target_probabilities(0)
        assert tp[0] > tp[1], "Incorrect probabilities."
        assert tp[1] == tp[2], "Incorrect probabilities."
        assert tp[2] > tp[3], "Incorrect probabilities."

    def test_initialization(self):
        n = 5
        g = TestPreferentialAttachment.create_graph(n=n)
        pa = PreferentialAttachment(graph=g, n=n)

        tp = pa.get_target_probabilities(2)
        assert len(tp) == n, "Incorrect number of targets."
        assert tp[0] > tp[1], "Incorrect probabilities."

    def test_update_degree_by_link(self):
        n = 5
        g = TestPreferentialAttachment.create_graph(n=n)
        pa = PreferentialAttachment(graph=g, n=n)

        g.add_edge(1,3)

        tp = pa.get_target_probabilities(0)

        assert tp[0] == tp[1], "Incorrect probabilities."
        assert tp[2] == tp[3], "Incorrect probabilities."

    def test_invalid_graph(self):
        n_g = 5
        n_pa = 3
        g = TestPreferentialAttachment.create_graph(n=n_g)

        with pytest.raises(AssertionError):
            _ = PreferentialAttachment(graph=g, n=n_pa)

    def test_no_valid_links(self):
        pass
