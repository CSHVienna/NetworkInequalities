import numpy as np

from netin.multidim.generate.utils import make_composite_index
from netin.multidim.generate.utils import comp_index_to_integer

def composite_H(
    h_mtrx_lst,
    kind,
    p_d = None,
    alpha = None,
    ):

    ## Convert elements of h matrix to np.array just in case
    h_mtrx_lst = [np.array(h) for h in h_mtrx_lst]

    if kind == "one":
        assert np.abs(sum(p_d)-1.0) < 1e-10 ## Verify normalization
    
    if kind == "max":
        assert 0.0 <= alpha <= 1.0
    
    g_vec = [len(h) for h in h_mtrx_lst]

    G = 1
    for g in g_vec:
        G *= g

    H_mtrx = np.zeros((G, G)) + np.nan

    comp_indices = make_composite_index(g_vec)
    assert len(comp_indices[0]) == len(h_mtrx_lst)

    for i_vec in comp_indices:
        for j_vec in comp_indices:
            
            I = comp_index_to_integer(i_vec,g_vec)
            J = comp_index_to_integer(j_vec,g_vec)
            
            if kind == "any":
                H_mtrx[I,J] = composite_H_ij_any(i_vec,j_vec,h_mtrx_lst)
            elif kind == "all":
                H_mtrx[I,J] = composite_H_ij_all(i_vec,j_vec,h_mtrx_lst)
            elif kind == "one":
                H_mtrx[I,J] = composite_H_ij_one(i_vec,j_vec,h_mtrx_lst,p_d)
            elif kind == "max":
                H_mtrx[I,J] = composite_H_ij_max(i_vec,j_vec,h_mtrx_lst,alpha)
            elif kind == "min":
                H_mtrx[I,J] = composite_H_ij_min(i_vec,j_vec,h_mtrx_lst,alpha)
            elif kind == "hierarchy":
                H_mtrx[I,J] = composite_H_ij_hierarchy(i_vec,j_vec,h_mtrx_lst)
            else:
                raise ValueError(f"Interaction kind {kind} invalid.")
    assert not np.any(np.isnan(H_mtrx))
    return H_mtrx

def composite_H_ij_any(i_vec,j_vec,h_mtrx_lst):
    
    Hij = 1.0
    for d, h in enumerate(h_mtrx_lst):
        Hij *= (1.0-h[i_vec[d],j_vec[d]])
    
    return 1.0 - Hij

def composite_H_ij_all(i_vec,j_vec,h_mtrx_lst):
    
    Hij = 1.0
    for d, h in enumerate(h_mtrx_lst):
        Hij *= (h[i_vec[d],j_vec[d]])
    
    return Hij

def composite_H_ij_one(i_vec,j_vec,h_mtrx_lst,p_d):
    
    Hij = 0.0
    for d, h in enumerate(h_mtrx_lst):
        Hij += p_d[d]*h[i_vec[d],j_vec[d]]
    
    return Hij

def composite_H_ij_max(i_vec,j_vec,h_mtrx_lst,alpha):

    D = 1.0*len(h_mtrx_lst)
    h_max = max([h[i_vec[d],j_vec[d]] for d,h in enumerate(h_mtrx_lst)])
    Hij = alpha*h_max

    for d, h in enumerate(h_mtrx_lst):
        Hij += (1.0-alpha) * h[i_vec[d],j_vec[d]] / D
    
    return Hij

def composite_H_ij_min(i_vec,j_vec,h_mtrx_lst,alpha):

    D = 1.0*len(h_mtrx_lst)
    h_min = min([h[i_vec[d],j_vec[d]] for d,h in enumerate(h_mtrx_lst)])
    Hij = alpha*h_min

    for d, h in enumerate(h_mtrx_lst):
        Hij += (1.0-alpha) * h[i_vec[d],j_vec[d]] / D
    
    return Hij

def composite_H_ij_hierarchy(i_vec,j_vec,h_mtrx_lst):

    for d, h in enumerate(h_mtrx_lst):
        if i_vec[d] != j_vec[d]:
            indx = list(i_vec[:d])
            indx.extend([i_vec[d],j_vec[d]])
            indx = tuple(indx)
            Hij = h[indx]
            break ## Only the first dimension with different group membership is considered
        elif d == len(h_mtrx_lst) -1:
            indx = list(i_vec[:d])
            indx.extend([i_vec[d],j_vec[d]])
            indx = tuple(indx)
            Hij = h[indx]
            break

    return Hij