import numpy as np

from netin.multidim.generate.utils import make_composite_index
from netin.multidim.generate.utils import comp_index_to_integer

##############################################################################
## Composite inter-group link metrics
##############################################################################

def comp_inter_group_ties(
    G,
    comp_pop_frac_tnsr,
    # m
    ):
    g_vec = comp_pop_frac_tnsr.shape
    comp_indices = make_composite_index(g_vec)
    num_groups = len(comp_indices)
    count_mtrx = np.zeros((num_groups,num_groups))
    I_to_ivec = {comp_index_to_integer(ivec,g_vec):ivec for ivec in comp_indices}
    ## Simple inter-group counts - GIVES SAME RESULT AS THE OTHER METHOD (simple_inter_group_ties)
    # simple_counts = [np.zeros((g,g)) for g in g_vec]
    for e,(o,d) in enumerate(G.edges()):
        
        i_type = G.nodes[o]["attr"]
        j_type = G.nodes[d]["attr"]

        I = comp_index_to_integer(i_type,g_vec)
        J = comp_index_to_integer(j_type,g_vec)

        count_mtrx[I,J] += 1

        ## Simple inter-group counts - GIVES SAME RESULT AS THE OTHER METHOD (simple_inter_group_ties)
        # for d, i in enumerate(i_type):
            # j = j_type[d]
            # simple_counts[d][i,j] += 1

    # return count_mtrx, I_to_ivec, simple_counts - GIVES SAME RESULT AS THE OTHER METHOD (simple_inter_group_ties)
    return count_mtrx, I_to_ivec

##############################################################################
## Simple inter-group link metrics
##############################################################################

def simple_inter_group_ties(count_mtrx,I_to_ivec,g_vec):
    ## 2022-10-25 changed input comp_pop_frac_tnsr by g_vec directly
    ## TO DO: Remove unused I_to_ivec parameter or implement some sort
    ## of test to assess the correctness of count_mtrx or something
    count_mtrx_lst = [np.zeros((g,g)) for g in g_vec]
    comp_indices = make_composite_index(g_vec)
    for ivec in comp_indices:
        I = comp_index_to_integer(ivec,g_vec)
        for jvec in comp_indices:
            J = comp_index_to_integer(jvec,g_vec)
            counts = count_mtrx[I,J]
            for d,i_d in enumerate(ivec):
                j_d = jvec[d]
                count_mtrx_lst[d][i_d,j_d] += counts
    return count_mtrx_lst

##############################################################################
## Composite groups population counts (population abundance tensor)
##############################################################################

def comp_group_cnt_tnsr(pop_cnt_vec,g_vec):
    comp_indices = make_composite_index(g_vec)
    pop_cnt_tnsr = np.zeros(g_vec)
    pop_cnt_lst = [np.zeros(g) for g in g_vec]
    for ivec in comp_indices:
        I = comp_index_to_integer(ivec,g_vec)
        counts = pop_cnt_vec[I]
        pop_cnt_tnsr[tuple(ivec)] = counts
        for d, i_d in enumerate(ivec):
            pop_cnt_lst[d][i_d] += counts
    return pop_cnt_tnsr, pop_cnt_lst