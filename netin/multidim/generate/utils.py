import itertools
import numpy as np

def make_composite_index(g_vec):
    assert all(g_vec)>0
    elems = [list(range(g)) for g in g_vec]
    comp_indices = itertools.product(*elems)
    return list(comp_indices)

def comp_index_to_integer(i_vec,g_vec):
    I = 0
    D = len(g_vec)
    for d, g in enumerate(g_vec):
        g_prod = 1
        for delta in range(d+1,D):
            g_prod *= g_vec[delta]
        I += i_vec[d]*g_prod
    return I

def get_num_multi_groups(g_vec):
    assert np.array(g_vec).ndim == 1
    G = 1
    for g in g_vec:
        G *= g
    return G