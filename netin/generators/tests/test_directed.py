from netin import DH
from netin import DPA
from netin import DPAH
from netin.utils import constants as const


class TestDiGraph(object):

    def test_patch_case_dpa(self):
        n = 200
        d = 0.01
        f_m = 0.1
        plo_M = 2.0
        plo_m = 2.0
        seed = 5678
        g = DPA(n=n, d=d, f_m=f_m, plo_M=plo_M, plo_m=plo_m, seed=seed)
        g.generate()
        c1 = g.is_directed()
        c2 = g.number_of_nodes() == n
        c3 = g.number_of_edges() == g.get_expected_number_of_edges()
        c4 = g.calculate_fraction_of_minority() == f_m
        c5 = g.get_model_name() == const.DPA_MODEL_NAME
        assert c1 and c2 and c3 and c4 and c5, "Incorrect directed parameters."

    def test_patch_case_dh(self):
        n = 200
        d = 0.01
        f_m = 0.1
        h_MM = 0.9
        h_mm = 0.9
        plo_M = 2.0
        plo_m = 2.0
        seed = 5678
        g = DH(n=n, d=d, f_m=f_m, h_MM=h_MM, h_mm=h_mm, plo_M=plo_M, plo_m=plo_m, seed=seed)
        g.generate()
        c1 = g.is_directed()
        c2 = g.number_of_nodes() == n
        c3 = g.number_of_edges() == g.get_expected_number_of_edges()
        c4 = g.calculate_fraction_of_minority() == f_m
        c5 = g.get_model_name() == const.DH_MODEL_NAME
        assert c1 and c2 and c3 and c4 and c5, "Incorrect directed parameters."

    def test_patch_case_dpah(self):
        n = 200
        d = 0.01
        f_m = 0.1
        h_MM = 0.9
        h_mm = 0.9
        plo_M = 2.0
        plo_m = 2.0
        seed = 5678
        g = DPAH(n=n, d=d, f_m=f_m, h_MM=h_MM, h_mm=h_mm, plo_M=plo_M, plo_m=plo_m, seed=seed)
        g.generate()
        c1 = g.is_directed()
        c2 = g.number_of_nodes() == n
        c3 = g.number_of_edges() == g.get_expected_number_of_edges()
        c4 = g.calculate_fraction_of_minority() == f_m
        c5 = g.get_model_name() == const.DPAH_MODEL_NAME
        assert c1 and c2 and c3 and c4 and c5, "Incorrect directed parameters."

    def test_patch_case_all(self):
        n = 200
        d = 0.01
        f_m = 0.1
        h_MM = 0.9
        h_mm = 0.9
        plo_M = 2.0
        plo_m = 2.0
        seed = 1234
        g_dpa = DPA(n=n, d=d, f_m=f_m, plo_M=plo_M, plo_m=plo_m, seed=seed)
        g_dh = DH(n=n, d=d, f_m=f_m, h_MM=h_MM, h_mm=h_mm, plo_M=plo_M, plo_m=plo_m, seed=seed)
        g_dpah = DPAH(n=n, d=d, f_m=f_m, h_MM=h_MM, h_mm=h_mm, plo_M=plo_M, plo_m=plo_m, seed=seed)

        g_dpa.generate()
        g_dh.generate()
        g_dpah.generate()

        c1 = g_dpa.number_of_nodes() == g_dh.number_of_nodes() == g_dpah.number_of_nodes() == n
        c2 = g_dpa.calculate_fraction_of_minority() == g_dh.calculate_fraction_of_minority() == g_dpah.calculate_fraction_of_minority() == f_m
        c3 = g_dpa.get_expected_number_of_edges() == g_dh.get_expected_number_of_edges() == g_dpah.get_expected_number_of_edges()

        c4 = g_dpa.get_model_name() == const.DPA_MODEL_NAME
        c5 = g_dh.get_model_name() == const.DH_MODEL_NAME
        c6 = g_dpah.get_model_name() == const.DPAH_MODEL_NAME
        assert c1 and c2 and c3 and c4 and c5 and c6, "Incorrect directed parameters."
