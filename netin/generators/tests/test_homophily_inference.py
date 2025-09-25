from netin import PA
from netin import PAH
from netin import PATC
from netin import PATCH
from netin.utils import constants as const
import netin
import numpy as np


class TestHomophilyInference(object):

    def test_patch_case_pah(self):
        n = 1000
        k = 2
        f_m = 0.1
        h_MM = 0.7
        h_mm = 0.7
        # seed = 1234
        # g = netin.PAH(n=n, k=k, f_m=f_m, h_MM=h_MM, h_mm=h_mm, seed=seed)
        g = netin.PAH(n=n, k=k, f_m=f_m, h_MM=h_MM, h_mm=h_mm)
        g.generate()
        inferred_homophily = g.homophily_inference()

        c1 = np.abs( inferred_homophily - h_MM ) < 0.1

        assert c1, "Incorrect value of homophily inference"

    def test_patch_case_pah_neutral(self):
        n = 1000
        k = 2
        f_m = 0.1
        h_MM = 0.5  # Changed from 0.7 to 0.5
        h_mm = 0.5  # Changed from 0.7 to 0.5
        g = netin.PAH(n=n, k=k, f_m=f_m, h_MM=h_MM, h_mm=h_mm)
        g.generate()
        inferred_homophily = g.homophily_inference()

        c1 = np.abs(inferred_homophily - h_MM) < 0.1

        assert c1, "Incorrect value of homophily inference"


    def test_patch_case_pa(self):
        n = 2000  # Increased network size
        k = 4  # Increased average degree
        f_m = 0.15
        # h_MM = 0.8
        # h_mm = 0.8
        g = netin.PA(n=n, k=k, f_m=f_m)
        g.generate()
        inferred_homophily = g.homophily_inference()

        c1 = np.abs(inferred_homophily - 0.5) < 0.1

        assert c1, "Incorrect value of homophily inference"


    def test_patch_case_patch_hetero(self):
        n = 2000  # Increased network size
        k = 4  # Increased average degree
        f_m = 0.15
        h_MM = 0.3
        h_mm = 0.3
        g = netin.PATCH(n=n, k=k, f_m=f_m, h_MM=h_MM, h_mm=h_mm, tc = 0.0)
        g.generate()
        inferred_homophily = g.homophily_inference()

        c1 = np.abs(inferred_homophily - h_mm) < 0.1

        assert c1, "Incorrect value of homophily inference"
