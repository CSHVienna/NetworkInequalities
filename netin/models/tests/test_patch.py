import pytest
from collections import defaultdict

import numpy as np
import scipy as sc

from collections import Counter
from itertools import product

from ..patch_model import PATCHModel, CompoundLFM
from ..pah_model import PAHModel
from ...graphs.graph import Graph
from ...graphs.directed import DiGraph
from ...utils.event_handling import Event
from ...utils.constants import CLASS_ATTRIBUTE

class TestPATCHModel:
    @staticmethod
    def create_model(
            n=5, f_m=.3, k=2,
            p_tc=.8,
            lfm_local=CompoundLFM.PAH,
            lfm_global=CompoundLFM.PAH,
            lfm_params={"h_mm": .8, "h_MM": .8},
            seed=123) -> PATCHModel:
        model = PATCHModel(
            n=n, f_m=f_m, k=k, p_tc=p_tc,
            lfm_local=lfm_local, lfm_global=lfm_global,
            lfm_params=lfm_params,
            seed=seed)
        return model

    @staticmethod
    def count_edge_types(g: Graph) -> Counter:
        counter = Counter()
        for source, target in g.edges():
            if g.get_node_class(CLASS_ATTRIBUTE)[source] == g.get_node_class(CLASS_ATTRIBUTE)[target]:
                counter["in_group"] += 1
            else:
                counter["out_group"] += 1
        return counter

    def test_lfm_assignments(self):
        h_params = dict(h_mm = .8, h_MM = .8)
        for lfm_l, lfm_g in product((CompoundLFM.UNIFORM, CompoundLFM.HOMOPHILY, CompoundLFM.PAH), repeat=2):
            model = TestPATCHModel.create_model(
                lfm_local=lfm_l, lfm_global=lfm_g,
            lfm_params=h_params)
            assert model.lfm_local == lfm_l
            assert model.lfm_global == lfm_g

            model.simulate()

            if lfm_l != CompoundLFM.UNIFORM or lfm_g != CompoundLFM.UNIFORM:
                assert model.h is not None
                with pytest.raises(AssertionError):
                    m_fail = TestPATCHModel.create_model(
                        lfm_local=lfm_l,
                        lfm_global=lfm_g,
                        lfm_params=None)
                    m_fail.simulate()
            if CompoundLFM.PAH in (lfm_l, lfm_g):
                assert model.pa is not None


        with pytest.raises(AssertionError):
            _ = TestPATCHModel.create_model(
                lfm_local="test")
        with pytest.raises(AssertionError):
            _ = TestPATCHModel.create_model(
                lfm_global="test")

    def test_event_handling(self):
        model = TestPATCHModel.create_model(n=100)
        counter = Counter()

        def _inc_counter(lfm: str):
            counter[lfm] += 1

        model.register_event_handler(
            event=Event.TARGET_SELECTION_LOCAL,
            function=lambda: _inc_counter("local"))
        model.register_event_handler(
            event=Event.TARGET_SELECTION_GLOBAL,
            function=lambda: _inc_counter("global"))

        model.simulate()

        assert counter["local"] > 0
        assert counter["global"] > 0
        assert counter["local"] + counter["global"] == (model.n - model.k) * model.k

    def test_simulation(self):
        model = TestPATCHModel.create_model(n=750)
        model.simulate()

        assert model.graph is not None
        assert not model.graph.is_directed()
        assert len(model.graph) == model.n
        _sum_links = sum(model.graph.degree(v)\
                         for v in model.graph.nodes())
        assert (_sum_links // 2) == ((model.n - model.k) * model.k)\
            + ((model.k * (model.k - 1)) // 2)

        degrees = sorted(
            [model.graph.degree(v) for v in model.graph.nodes()])
        assert degrees[0] == model.k
        assert degrees[-1] >= 3 * model.k

        nodes_min = set(
            node\
                for node in model.graph.nodes()\
                    if model.graph.get_node_class("minority")[node])
        assert np.isclose(
            len(nodes_min), model.f_m * model.n, rtol=0.05)
        n_in_group, n_out_group = 0, 0
        for source, target in model.graph.edges():
            if model.graph.get_node_class("minority")[source]\
                == model.graph.get_node_class("minority")[target]:
                n_in_group += 1
            else:
                n_out_group += 1
        assert (n_in_group > 0) and (n_out_group > 0)
        assert n_in_group > n_out_group

    def test_preload_graph(self):
        n = 100
        n_g = 50
        model = TestPATCHModel.create_model(n=n)

        with pytest.raises(AssertionError):
            model.preload_graph(DiGraph())

        g_pre = Graph()
        for i in range(n_g):
            g_pre.add_node(i)
        g_pre.add_edge(0, 1)

        model.preload_graph(graph=g_pre)
        model.simulate()

        assert len(model.graph) == (n + n_g)
        _sum_links = sum(model.graph.degree(v)\
                         for v in model.graph.nodes()) // 2
        assert _sum_links == (1 + (n * model.k))

    def test_pah_reduction(self):
        n = 500
        n_iter = 100
        ratios_total = defaultdict(list)
        for seed in range(n_iter):
            g_patch = TestPATCHModel.create_model(
                n=n,
                p_tc=0.0,
                lfm_local=CompoundLFM.PAH, lfm_global=CompoundLFM.PAH,
                seed=seed)
            g_pah = PAHModel(
                n=g_patch.n, k=g_patch.k, f_m=g_patch.f_m,
                h_mm=g_patch.lfm_params["h_mm"],
                h_MM=g_patch.lfm_params["h_MM"],
                seed=seed
            )

            g_patch.simulate()
            g_pah.simulate()

            cnt_patch = TestPATCHModel.count_edge_types(g_patch.graph)
            cnt_pah = TestPATCHModel.count_edge_types(g_pah.graph)

            ratio_patch = cnt_patch["in_group"] / cnt_patch["out_group"]
            ratio_pah = cnt_pah["in_group"] / cnt_pah["out_group"]

            ratios_total["patch"].append(ratio_patch)
            ratios_total["pah"].append(ratio_pah)
        test = sc.stats.ttest_ind(ratios_total["patch"], ratios_total["pah"])
        assert test.pvalue >= 0.05
