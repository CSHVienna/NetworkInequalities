from netin.models import BarabasiAlbertModel
from netin.utils import constants as const


class TestUnDiGraph(object):

    def test_patch_case_pa(self):
        n = 200
        k = 2
        f_m = 0.1
        seed = 5678
        m = BarabasiAlbertModel(
            N=n, m=k, f=f_m, seed=seed)
        m.simulate()
        g=m.graph
        c1 = not g.is_directed()
        c2 = g.number_of_nodes() == n
        # c3 = g.calculate_minimum_degree() == k
        # c4 = g.calculate_fraction_of_minority() == f_m
        # c5 = g.model_name == const.PA_MODEL_NAME
        c6 = sum(k for _, k in g.degree()) == ((k*(k-1)) + ((n-k)*k*2))
        assert c1 and c2 and c6, "Incorrect undirected parameters."

    # def test_patch_case_pah(self):
    #     n = 200
    #     k = 2
    #     f_m = 0.1
    #     h_MM = 0.9
    #     h_mm = 0.1
    #     seed = 5678
    #     g = PAH(n=n, k=k, f_m=f_m, h_MM=h_MM, h_mm=h_mm, seed=seed)
    #     g.generate()
    #     c1 = not g.is_directed()
    #     c2 = g.number_of_nodes() == n
    #     c3 = g.calculate_minimum_degree() == k
    #     c4 = g.calculate_fraction_of_minority() == f_m
    #     c5 = g.model_name == const.PAH_MODEL_NAME
    #     c6 = sum(k for _, k in g.degree()) == ((k*(k-1)) + ((n-k)*k*2))
    #     assert c1 and c2 and c3 and c4 and c5 and c6, "Incorrect undirected parameters."

    # def test_case_patc(self):
    #     n = 200
    #     k = 2
    #     f_m = 0.1
    #     tc = 0.5
    #     seed = 5678
    #     g = PATC(n=n, k=k, f_m=f_m, tc=tc, seed=seed)
    #     g.generate()
    #     c1 = not g.is_directed()
    #     c2 = g.number_of_nodes() == n
    #     c3 = g.calculate_minimum_degree() == k
    #     c4 = g.calculate_fraction_of_minority() == f_m
    #     c5 = g.model_name == const.PATC_MODEL_NAME
    #     c6 = sum(k for _, k in g.degree()) == ((k*(k-1)) + ((n-k)*k*2))
    #     assert c1 and c2 and c3 and c4 and c5 and c6, "Incorrect undirected parameters."

    # def test_case_patch(self):
    #     n = 200
    #     k = 2
    #     f_m = 0.1
    #     h_MM = 0.1
    #     h_mm = 0.1
    #     tc = 1.0
    #     seed = 5678
    #     g = PATCH(n=n, k=k, f_m=f_m, h_MM=h_MM, h_mm=h_mm, tc=tc, seed=seed)
    #     g.generate()
    #     c1 = not g.is_directed()
    #     c2 = g.number_of_nodes() == n
    #     c3 = g.calculate_minimum_degree() == k
    #     c4 = g.calculate_fraction_of_minority() == f_m
    #     c5 = g.model_name == const.PATCH_MODEL_NAME
    #     c6 = sum(k for _, k in g.degree()) == ((k*(k-1)) + ((n-k)*k*2))
    #     assert c1 and c2 and c3 and c4 and c5 and c6, "Incorrect undirected parameters."

    # def test_case_all(self):
    #     n = 200
    #     k = 2
    #     f_m = 0.1
    #     h_MM = 0.5
    #     h_mm = 0.5
    #     tc = 0.0
    #     seed = 1234
    #     g_pa = PA(n=n, k=k, f_m=f_m, seed=seed)
    #     g_pah = PAH(n=n, k=k, f_m=f_m, h_MM=h_MM, h_mm=h_mm, seed=seed)
    #     g_patc = PATC(n=n, k=k, f_m=f_m, tc=tc, seed=seed)
    #     g_patch = PATCH(n=n, k=k, f_m=f_m, h_MM=h_MM, h_mm=h_mm, tc=tc, seed=seed)

    #     g_pa.generate()
    #     g_pah.generate()
    #     g_patc.generate()
    #     g_patch.generate()

    #     c1 = not g_pa.is_directed() and not g_pah.is_directed() and not g_patc.is_directed() and not g_patch.is_directed()
    #     c2 = g_pa.number_of_nodes() == g_pah.number_of_nodes() == g_patc.number_of_nodes() == g_patch.number_of_nodes() == n
    #     c3 = g_pa.calculate_minimum_degree() == g_pah.calculate_minimum_degree() == g_patc.calculate_minimum_degree() == g_patch.calculate_minimum_degree() == k
    #     c4 = g_pa.calculate_fraction_of_minority() == g_pah.calculate_fraction_of_minority() == g_patc.calculate_fraction_of_minority() == g_patch.calculate_fraction_of_minority() == f_m
    #     c5 = g_pa.get_expected_number_of_edges() == g_pah.get_expected_number_of_edges() == g_patc.get_expected_number_of_edges() == g_patch.get_expected_number_of_edges()
    #     c6 = g_pa.model_name == const.PA_MODEL_NAME
    #     c7 = g_pah.model_name == const.PAH_MODEL_NAME
    #     c8 = g_patc.model_name == const.PATC_MODEL_NAME
    #     c9 = g_patch.model_name == const.PATCH_MODEL_NAME
    #     assert c1 and c2 and c3 and c4 and c5 and c6 and c7 and c8 and c9, "Incorrect undirected parameters."
