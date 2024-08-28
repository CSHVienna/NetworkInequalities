import numpy as np
from netin.multidim.generate.utils import make_composite_index

def is_pop_frac_consistent(pop_fracs_lst,comp_pop_frac_tnsr):

    pop_fracs_lst = np.array(pop_fracs_lst)
    
    assert np.all(comp_pop_frac_tnsr>=0)
    assert np.all(comp_pop_frac_tnsr<=1)

    assert np.all(pop_fracs_lst>=0)
    assert np.all(pop_fracs_lst<=1)

    ## Check pop_fracs_lst ordered from smaller to larger populations
    ## We assume this for several computations, so it is better to
    ## be sure
    for pop_fracs in pop_fracs_lst:
        assert np.all(np.sort(pop_fracs) == pop_fracs)

    ## Check overall normalization
    for d, pop_frac in enumerate(pop_fracs_lst):
        if np.abs(np.sum(pop_frac) - 1.0) > 1e-10:
            print (f"Bad normalization in simple populations of dim {d}")
            return False

    if np.abs(np.sum(comp_pop_frac_tnsr) - 1.0) > 1e-10:
        print ("Bad overall normalization of composite population fractions.")
        return False

    ## Check marginals
    g_vec = [len(pop_fracs) for pop_fracs in pop_fracs_lst]
    D = len(pop_fracs_lst)
    comp_indices = make_composite_index(g_vec)
    for d, g in enumerate(g_vec):
        for i in range(g):
            
            fdi = pop_fracs_lst[d][i]

            ## As per https://stackoverflow.com/questions/12116830/numpy-slice-of-arbitrary-dimensions
            ## arr[(None,1,None,None)] == arr[:,1,:,:]
            indx = [slice(None)]*comp_pop_frac_tnsr.ndim
            indx[d] = i
            indx = tuple(indx)
            Fdi = np.sum(comp_pop_frac_tnsr[indx])

            if np.abs(fdi - Fdi) > 1e-13:
                print (f"Bad marginals in group {i} of dim {d}: {fdi} / {Fdi}")
                return False

    return True