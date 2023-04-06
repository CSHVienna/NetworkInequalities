import gc
import pytest

from netin import Graph
from netin.utils import constants as const

class TestPATCH(object):

    def test_patch_case_1(self):
        n = 200
        k = 2
        f_m = 0.1
        h_MM = 0.1
        h_mm = 0.1
        tc = 0.5
        seed = 1234
        g = Graph(n=n, k=k, f_m=f_m, h_MM=h_MM, h_mm=h_mm, tc=tc, seed=seed)
        g.generate()
        c1 = g.number_of_nodes() == n
        c2 = g.calculate_minimum_degree() == k
        c3 = g.calculate_fraction_of_minority() == f_m
        c4 = g.get_model_name() == const.PATCH_MODEL_NAME
        assert c1 and c2 and c3 and c4, "Incorrect graph parameters."

    def test_patch_case_2(self):
        n = 200
        k = 2
        f_m = 0.1
        h_MM = 0.9
        h_mm = 0.1
        tc = 0.5
        seed = 1234
        g = Graph(n=n, k=k, f_m=f_m, h_MM=h_MM, h_mm=h_mm, tc=tc, seed=seed)
        g.generate()
        c1 = g.number_of_nodes() == n
        c2 = g.calculate_minimum_degree() == k
        c3 = g.calculate_fraction_of_minority() == f_m
        c4 = g.get_model_name() == const.PATCH_MODEL_NAME
        assert c1 and c2 and c3 and c4, "Incorrect graph parameters."

    def test_patch_case_3(self):
        n = 200
        k = 2
        f_m = 0.1
        h_MM = 0.1
        h_mm = 0.9
        tc = 0.5
        seed = 1234
        g = Graph(n=n, k=k, f_m=f_m, h_MM=h_MM, h_mm=h_mm, tc=tc, seed=seed)
        g.generate()
        c1 = g.number_of_nodes() == n
        c2 = g.calculate_minimum_degree() == k
        c3 = g.calculate_fraction_of_minority() == f_m
        c4 = g.get_model_name() == const.PATCH_MODEL_NAME
        assert c1 and c2 and c3 and c4, "Incorrect graph parameters."

    def test_patch_case_4(self):
        n = 200
        k = 2
        f_m = 0.1
        h_MM = 0.1
        h_mm = 0.1
        tc = 1.0
        seed = 1234
        g = Graph(n=n, k=k, f_m=f_m, h_MM=h_MM, h_mm=h_mm, tc=tc, seed=seed)
        g.generate()
        c1 = g.number_of_nodes() == n
        c2 = g.calculate_minimum_degree() == k
        c3 = g.calculate_fraction_of_minority() == f_m
        c4 = g.get_model_name() == const.PATCH_MODEL_NAME
        assert c1 and c2 and c3 and c4, "Incorrect graph parameters."

    def test_patch_case_5(self):
        n = 200
        k = 2
        f_m = 0.1
        h_MM = 0.1
        h_mm = 0.1
        seed = 1234
        g = Graph(n=n, k=k, f_m=f_m, h_MM=h_MM, h_mm=h_mm, seed=seed)
        g.generate()
        c1 = g.number_of_nodes() == n
        c2 = g.calculate_minimum_degree() == k
        c3 = g.calculate_fraction_of_minority() == f_m
        c4 = g.get_model_name() == const.PAH_MODEL_NAME
        assert c1 and c2 and c3 and c4, "Incorrect graph parameters."

    def test_patch_case_6(self):
        n = 200
        k = 2
        f_m = 0.1
        tc = 0.2
        seed = 1234
        g = Graph(n=n, k=k, f_m=f_m, tc=tc, seed=seed)
        g.generate()
        c1 = g.number_of_nodes() == n
        c2 = g.calculate_minimum_degree() == k
        c3 = g.calculate_fraction_of_minority() == f_m
        c4 = g.get_model_name() == const.PATC_MODEL_NAME
        assert c1 and c2 and c3 and c4, "Incorrect graph parameters."

    def test_patch_case_7(self):
        n = 200
        k = 2
        f_m = 0.1
        seed = 1234
        g = Graph(n=n, k=k, f_m=f_m, seed=seed)
        g.generate()
        c1 = g.number_of_nodes() == n
        c2 = g.calculate_minimum_degree() == k
        c3 = g.calculate_fraction_of_minority() == f_m
        c4 = g.get_model_name() == const.PA_MODEL_NAME
        assert c1 and c2 and c3 and c4, "Incorrect graph parameters."

