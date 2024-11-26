from typing import Dict, Tuple
from netin.graphs import Graph
from netin.utils.constants import CLASS_ATTRIBUTE
from netin.models import PAHModel

from sympy import Eq
from sympy import solve
from sympy import symbols
import powerlaw

import numpy as np

def cmp_new(a_h: np.ndarray, a_pa: np.ndarray) -> np.ndarray:
    a_p = (a_h / a_h.sum()) * (a_pa / a_pa.sum())
    return a_p / a_p.sum()

def cmp_old(a_h: np.ndarray, a_pa: np.ndarray) -> np.ndarray:
    a_p = a_h * a_pa
    return a_p / a_p.sum()

def calculate_edge_type_counts(graph: Graph) -> Dict[str, int]:
    minority_nodes = graph.get_node_class(CLASS_ATTRIBUTE).get_class_values()
    counter = {
        "MM": 0,
        "Mm": 0,
        "mM": 0,
        "mm": 0
    }
    for source, target in graph.edges():
        counter[f"{minority_nodes[source]}{minority_nodes[target]}"] += 1
    return counter

def calculate_degree_powerlaw_exponents(graph: Graph) -> Tuple[float, float]:
        degrees = graph.degrees()
        minority_nodes = graph.get_node_class(CLASS_ATTRIBUTE)

        dM = degrees[minority_nodes.get_majority_mask()]
        dm = degrees[minority_nodes.get_minority_mask()]

        fit_M = powerlaw.Fit(data=dM, discrete=True, xmin=min(dM), xmax=max(dM), verbose=False)
        fit_m = powerlaw.Fit(data=dm, discrete=True, xmin=min(dm), xmax=max(dm), verbose=False)

        pl_M = fit_M.power_law.alpha
        pl_m = fit_m.power_law.alpha
        return pl_M, pl_m

def infer_homophily_values(graph: Graph) -> Tuple[float, float]:
    """
    Infers analytically the homophily values for the majority and minority classes.

    Returns
    -------
    h_MM: float
        homophily within majority group

    h_mm: float
        homophily within minority group
    """

    f_m = np.mean(graph.get_node_class(CLASS_ATTRIBUTE))
    f_M = 1 - f_m

    e = calculate_edge_type_counts(graph)
    e_MM = e['MM']
    e_mm = e['mm']
    M = e['MM'] + e['mm'] + e['Mm'] + e['mM']

    p_MM = e_MM / M
    p_mm = e_mm / M

    pl_M, pl_m = calculate_degree_powerlaw_exponents(graph)
    b_M = -1 / (pl_M + 1)
    b_m = -1 / (pl_m + 1)

    # equations
    hmm, hMM, hmM, hMm = symbols('hmm hMM hmM hMm')
    eq1 = Eq((f_m * f_m * hmm * (1 - b_M)) / ((f_m * hmm * (1 - b_M)) + (f_M * hmM * (1 - b_m))), p_mm)
    eq2 = Eq(hmm + hmM, 1)

    eq3 = Eq((f_M * f_M * hMM * (1 - b_m)) / ((f_M * hMM * (1 - b_m)) + (f_m * hMm * (1 - b_M))), p_MM)
    eq4 = Eq(hMM + hMm, 1)

    solution = solve((eq1, eq2, eq3, eq4), (hmm, hmM, hMM, hMm))
    h_MM, h_mm = solution[hMM], solution[hmm]
    return h_MM, h_mm

def main():
    n = 2000
    k = 2
    f_m = .2
    h = .8

    a_h_M_infer, a_h_m_infer = [], []
    for i in range(50):
        model = PAHModel(n=n, f_m=f_m, k=k, h_mm=h, h_MM=h, seed=i)
        g = model.simulate()

        print(calculate_edge_type_counts(g))
        h_M_inf, h_m_inf = infer_homophily_values(g)

        a_h_M_infer.append(h_M_inf)
        a_h_m_infer.append(h_m_inf)

    a_h_M_infer = np.asarray(a_h_M_infer)
    a_h_m_infer = np.asarray(a_h_m_infer)

    print("Minority")
    print(np.mean(a_h_m_infer))

    print("Majority")
    print(np.mean(a_h_M_infer))

    a_h = np.asarray([.2, .2, .8, .2, .8])
    a_pa = np.asarray([4, 8, 2, 2, 3])
    print(
        cmp_new(a_h, a_pa),
        cmp_old(a_h, a_pa)
    )

if __name__ == "__main__":
    main()
