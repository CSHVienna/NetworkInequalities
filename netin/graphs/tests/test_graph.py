from typing import Dict

from ..graph import Graph
from ..directed import DiGraph
from ..event import Event

class TestGraph(object):
    @staticmethod
    def assert_link_add_before(
            node_u: int, node_v: int,
            link_u: int, link_v: int,
            d: Dict[str, str], graph: Graph):
        print((
            f"Triggered before-link-addition event\n"
            f"Nodes u={link_u} and v={link_v}"))
        assert(node_u == link_u and node_v == link_v)
        assert(not graph.has_edge(link_u, link_v))
        d["assert_link_add_before"] = True

    @staticmethod
    def assert_link_add_after(
            node_u: int, node_v: int,
            link_u: int, link_v: int,
            d: Dict[str, bool], graph: Graph):
        print((
            f"Triggered after-link-addition event\n"
            f"Nodes u={link_u} and v={link_v}"))
        assert(node_u == link_u and node_v == link_v)
        assert(graph.has_edge(link_u, link_v))
        if graph.is_directed():
            assert(not graph.has_edge(link_v, link_u))
        else:
            assert(graph.has_edge(link_v, link_u))
        d["assert_link_add_after"] = True

    def test_nodes_and_edge_addition(self):
        g = Graph()
        n = 10
        for i in range(1,n):
            for j in range(i):
                g.add_edge(i, j)
        assert(g.number_of_nodes() == n), "Incorrect number of nodes."
        assert(g.number_of_edges() == n*(n-1)//2), "Incorrect number of edges."

    def test_inheritance(self):
        ug = Graph()
        dg = DiGraph()
        assert isinstance(dg, Graph), "Directed graph is not an instance of `Graph`."
        assert not isinstance(ug, DiGraph), "Undirected graph is an instance of `DiGraph`."

    def test_directedness(self):
        ug = Graph()
        dg = DiGraph()
        d = {}
        source, target = 0, 1

        assert not ug.is_directed(), "Undirected graph is directed."
        assert dg.is_directed(), "Directed graph is undirected."

        dg.register_event_handler(
            event=Event.LINK_ADD_AFTER,
            function=lambda source_eh, target_eh:\
                TestGraph.assert_link_add_after(source, target, source_eh, target_eh, d, dg))
        dg.add_edge(source, target)

        assert(("assert_link_add_after" in d) and d["assert_link_add_after"]), "Function `assert_link_add_after` not called"

    def test_events(self):
        g = Graph()
        node_u, node_v = 1, 2
        d = {}

        g.register_event_handler(
            event=Event.LINK_ADD_BEFORE,
            function=lambda link_u, link_v:\
                TestGraph.assert_link_add_before(node_u, node_v, link_u, link_v, d, g)
        )

        g.register_event_handler(
            event=Event.LINK_ADD_AFTER,
            function=lambda link_u, link_v:\
                TestGraph.assert_link_add_after(node_u, node_v, link_u, link_v, d, g)
        )

        g.add_edge(node_u, node_v)

        assert(("assert_link_add_before" in d) and d["assert_link_add_before"]), "Function `assert_link_add_before` not called"
        assert(("assert_link_add_after" in d) and d["assert_link_add_after"]), "Function `assert_link_add_after` not called"
