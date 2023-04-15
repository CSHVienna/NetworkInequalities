import networkx as nx
import pytest

from netin import DPAH
from netin.utils import constants as const


class TestDPAH(object):

    def test_dpah_case_1(self):
        n = 200
        d = 0.1
        f_m = 0.1
        plo_M = 2.0
        plo_m = 2.0
        h_MM = 0.1
        h_mm = 0.1
        seed = 1234
        g = DPAH(n=n, d=d, f_m=f_m, plo_M=plo_M, plo_m=plo_m, h_MM=h_MM, h_mm=h_mm, seed=seed)
        g.generate()
        c1 = g.is_directed()
        c2 = g.number_of_nodes() == n
        c3 = nx.density(g) == d
        c4 = g.calculate_fraction_of_minority() == f_m
        c5 = g.get_model_name() == const.DPAH_MODEL_NAME
        assert c1 and c2 and c3 and c4 and c5, "Incorrect directed parameters."

    def test_dpah_case_2(self):
        n = 200
        d = 0.1
        f_m = 0.1
        plo_M = 2.0
        plo_m = 2.0
        h_MM = 0.9
        h_mm = 0.1
        seed = 1234
        g = DPAH(n=n, d=d, f_m=f_m, plo_M=plo_M, plo_m=plo_m, h_MM=h_MM, h_mm=h_mm, seed=seed)
        g.generate()
        c1 = g.is_directed()
        c2 = g.number_of_nodes() == n
        c3 = nx.density(g) == d
        c4 = g.calculate_fraction_of_minority() == f_m
        c5 = g.get_model_name() == const.DPAH_MODEL_NAME
        assert c1 and c2 and c3 and c4 and c5, "Incorrect directed parameters."

    def test_dpah_case_3(self):
        n = 200
        d = 0.1
        f_m = 0.1
        plo_M = 2.0
        plo_m = 2.0
        h_MM = 0.1
        h_mm = 0.9
        seed = 1234
        g = DPAH(n=n, d=d, f_m=f_m, plo_M=plo_M, plo_m=plo_m, h_MM=h_MM, h_mm=h_mm, seed=seed)
        g.generate()
        c1 = g.is_directed()
        c2 = g.number_of_nodes() == n
        c3 = nx.density(g) == d
        c4 = g.calculate_fraction_of_minority() == f_m
        c5 = g.get_model_name() == const.DPAH_MODEL_NAME
        assert c1 and c2 and c3 and c4 and c5, "Incorrect directed parameters."

    def test_dpah_case_4(self):
        n = 200
        d = 0.1
        f_m = 0.1
        plo_M = 3.0
        plo_m = 1.0
        h_MM = 0.1
        h_mm = 0.1
        seed = 1234
        with pytest.raises(ValueError, match="Value is out of range."):
            g = DPAH(n=n, d=d, f_m=f_m, plo_M=plo_M, plo_m=plo_m, h_MM=h_MM, h_mm=h_mm, seed=seed)
            g.generate()

    def test_dpah_case_5(self):
        n = 200
        d = 0.1
        f_m = 0.1
        plo_M = 1.0
        plo_m = 3.0
        h_MM = 0.1
        h_mm = 0.1
        seed = 1234
        with pytest.raises(ValueError, match="Value is out of range."):
            g = DPAH(n=n, d=d, f_m=f_m, plo_M=plo_M, plo_m=plo_m, h_MM=h_MM, h_mm=h_mm, seed=seed)
            g.generate()

    def test_dpah_case_6(self):
        n = 200
        d = 0.1
        f_m = 0.1
        plo_M = 1.0
        plo_m = 3.0
        h_MM = 0.1
        h_mm = 0.1
        seed = 1234
        with pytest.raises(TypeError, match="missing 1 required positional argument: 'plo_M'"):
            _ = DPAH(n=n, d=d, f_m=f_m, plo_m=plo_m, h_MM=h_MM, h_mm=h_mm, seed=seed)

    def test_dpah_case_7(self):
        n = 200
        d = 0.1
        f_m = 0.1
        plo_M = 1.0
        plo_m = 3.0
        h_MM = 0.1
        h_mm = 0.1
        seed = 1234
        with pytest.raises(TypeError, match="missing 1 required positional argument: 'f_m'"):
            _ = DPAH(n=n, d=d, plo_M=plo_M, plo_m=plo_m, h_MM=h_MM, h_mm=h_mm, seed=seed)

    def test_dpah_case_8(self):
        n = 200
        d = 0.1
        f_m = 0.1
        plo_M = 1.0
        plo_m = 3.0
        h_MM = 0.1
        h_mm = 0.1
        seed = 1234
        with pytest.raises(TypeError, match="missing 2 required positional arguments: 'plo_M' and 'h_mm'"):
            _ = DPAH(n=n, d=d, f_m=f_m, plo_m=plo_m, h_MM=h_MM, seed=seed)

    def test_dpah_case_9(self):
        n = 200
        d = 0.1
        f_m = 0.1
        plo_M = 1.0
        plo_m = 3.0
        h_MM = 0.1
        h_mm = 0.1
        seed = 1234
        with pytest.raises(TypeError,
                           match="missing 7 required positional arguments: 'n', 'd', 'f_m', 'plo_M', 'plo_m', 'h_MM', and 'h_mm'"):
            _ = DPAH(seed=seed)
