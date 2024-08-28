import numpy as np

from netin.multidim.generate.utils import make_composite_index
from netin.multidim.generate.utils import comp_index_to_integer
from netin.multidim.theory.multidimensional_preferences import composite_H

##############################################################################
## Erdos-Renyi like (or block-model like) N-dimensional theoretical number of 
## inter-group links
##############################################################################

def ER_inter_group_links_theor(
    h_mtrx_lst,
    comp_pop_frac_tnsr,
    kind,
    p_d = None,
    alpha = None
    ):
    H_theor = composite_H(
        h_mtrx_lst,
        kind,
        p_d = p_d,
        alpha = alpha,
        )

    g_vec = comp_pop_frac_tnsr.shape

    comp_indices = make_composite_index(g_vec)
    assert len(comp_indices[0]) == len(h_mtrx_lst)

    G = len(comp_indices)
    M_theor = np.zeros((G,G))+np.nan
    for i_vec in comp_indices:
        I = comp_index_to_integer(i_vec,g_vec)
        for j_vec in comp_indices:
            J = comp_index_to_integer(j_vec,g_vec)
            F_I = comp_pop_frac_tnsr[tuple(i_vec)]
            F_J = comp_pop_frac_tnsr[tuple(j_vec)]
            M_theor[I,J] = F_I*F_J*H_theor[I,J]

    return M_theor/np.sum(M_theor), H_theor

##############################################################################
## 1D Theoretical number of inter-group links
##############################################################################

def ER_1D_solve_h_mtrx_dens(M_cnts_dir,pop_cnts,get_CI = False):
#     E = np.sum(M_cnts_dir)
#     N = np.sum(pop_cnts)
#     rho = E/N**2.0 ## Approximation of E / (N (N-1))

    h_mtrx = np.zeros_like(M_cnts_dir)
    if get_CI:
        h_mtrx_u = np.zeros_like(M_cnts_dir)
        h_mtrx_l = np.zeros_like(M_cnts_dir)
    g = M_cnts_dir.shape[0]
    for i in range(g):
        for j in range(g):
            if pop_cnts[i]*pop_cnts[j] == 0:
                ## Enforce h_mtrx[i,j] = 0 instead of np.nan when it is 0/0
                h_mtrx[i,j] = 0
            else:
                
                h_mtrx[i,j] = M_cnts_dir[i,j] / (pop_cnts[i]*pop_cnts[j])
                
                if get_CI:
                    l,u = proportion_confint(
                        M_cnts_dir[i,j],
                        pop_cnts[i]*pop_cnts[j],
                        method="wilson")
                    h_mtrx_l[i,j] = l
                    h_mtrx_u[i,j] = u
    
    if get_CI:
        return h_mtrx, h_mtrx_l, h_mtrx_u
    else:
        return h_mtrx