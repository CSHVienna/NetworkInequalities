from typing import Dict

import networkx as nx
import numpy as np

from ..graph import Graph
from ..directed import DiGraph
from ...utils.event_handling import Event

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
        g.add_node(0)
        n = 10
        for i in range(1,n):
            g.add_node(i)
            for j in range(i):
                g.add_edge(i, j)
        assert(len(g) == n), "Incorrect number of nodes."
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

        for node in source, target:
            ug.add_node(node)
            dg.add_node(node)

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
        for node in node_u, node_v:
            g.add_node(node)

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

    def test_n_edges(self):
        g = Graph()
        n = 10
        for i in range(n):
            g.add_node(i)
            for j in range(i):
                g.add_edge(i, j)
        number_of_edges = g.number_of_edges()
        degree_sum = sum(g.degree(v) for v in g.nodes()) // 2
        edges_theory = n*(n-1)//2
        edges_iterator = sum(1 for edge in g.edges())
        assert number_of_edges == degree_sum, "Incorrect number of edges."
        assert degree_sum == edges_theory, "Incorrect number of edges."
        assert edges_theory == edges_iterator, "Incorrect number of edges."

    def test_to_nx_conversion(self):
        g = Graph()
        n = 10
        for i in range(n):
            g.add_node(i)
            for j in range(i):
                g.add_edge(i, j)
        g.set_node_class("minority", {i: i%2 for i in range(n)})

        nx_g = g.to_nxgraph()
        assert isinstance(nx_g, nx.Graph), "Conversion to NetworkX graph failed."
        assert len(nx_g) == n, "Incorrect number of nodes."
        assert nx_g.number_of_edges() == n*(n-1)//2, "Incorrect number of edges."
        for i in range(n):
            assert nx_g.nodes[i]["minority"] == i%2, "Incorrect node class."
            for j in range(i):
                assert nx_g.has_edge(i, j), "Incorrect edge."

    def test_from_nx_conversion(self):
        nx_g = nx.Graph()
        n = 10
        for i in range(n):
            nx_g.add_node(i)
            for j in range(i):
                nx_g.add_edge(i, j)
        nx.set_node_attributes(nx_g, {i: i%2 for i in range(n)}, "minority")

        node_ids, g = Graph.from_nxgraph(nx_g, node_attributes_names=["minority"])
        assert len(g) == n, "Incorrect number of nodes."
        assert len(node_ids) == n, "Incorrect number of nodes in mapping."
        assert np.all(node_ids == np.arange(n)), "Incorrect node mapping."
        assert g.number_of_edges() == n*(n-1)//2, "Incorrect number of edges."
        for i in range(n):
            assert g.get_node_class("minority")[i] == i%2, "Incorrect node class."
            for j in range(i):
                assert g.has_edge(i, j), "Incorrect edge."

    def test_from_nx_conv_custom_labels(self):
        nodes = np.asarray(["a", "b", "c", "d", "e"][::-1])
        edges = [("a", "b"), ("b", "c"), ("c", "d"), ("d", "e")]

        nx_g = nx.Graph()
        for node in nodes:
            nx_g.add_node(node)
        for edge in edges:
            nx_g.add_edge(*edge)
        nx.set_node_attributes(nx_g, {node: int(node in ("d", "e")) for node in nodes}, "minority")

        node_ids, g = Graph.from_nxgraph(
            nx_g, sort_node_labels=False, node_attributes_names=["minority"])
        assert np.all(nodes == node_ids), "Incorrect node mapping."
        minority = g.get_node_class("minority")
        for i, min_i in enumerate(minority):
            assert min_i == (node_ids[i] in ("d", "e")), "Incorrect node class."

        node_ids_sort, g_sort = Graph.from_nxgraph(
            nx_g, sort_node_labels=True, node_attributes_names=["minority"])
        assert np.all(nodes[::-1] == node_ids_sort), "Incorrect node mapping."
        minority_sort = g_sort.get_node_class("minority")
        for i, min_i in enumerate(minority_sort):
            assert min_i == (node_ids_sort[i] in ("d", "e")), "Incorrect node class."
