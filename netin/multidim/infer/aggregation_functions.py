import numpy as np

from netin.multidim.generate.utils import get_num_multi_groups
from netin.multidim.generate.utils import make_composite_index
from netin.multidim.infer import aux_functions

def product_f(x,g_vec):
    mat_list =  aux_functions.vec_to_mat_list(x,g_vec)
    return aux_functions.nested_f(mat_list,np.kron)

def mean_f(x,g_vec):
    mat_list = aux_functions.vec_to_mat_list(x,g_vec)
    J_list = [np.ones((i,i)) for i in g_vec]
    list_of_lists = [list(J_list[:i]) + [mat_list[i]] + list(J_list[i+1:]) for i in range(len(g_vec))]
    list_of_kron = [aux_functions.nested_f(X,np.kron) for X in list_of_lists]
    return aux_functions.nested_f(list_of_kron, lambda x_1,x_2: x_1+x_2)/len(g_vec)

def at_least_one_f(x,g_vec):
    tot_dim = int(np.prod(g_vec))
    x_list = aux_functions.vec_to_mat_list(x,g_vec)
    mat_list = [np.ones((g_vec[i],g_vec[i])) - x_list[i] for i in range(len(g_vec))]
    return np.ones((tot_dim,tot_dim)) - aux_functions.nested_f(mat_list,np.kron)

def cross_dimensional_product_f(x,g_vec):
    mat_list = aux_functions.vec_to_mat_dict_cross_one_dimensional(x,g_vec)
    n_elem_H = get_num_multi_groups(g_vec)
    H = np.ones((n_elem_H,n_elem_H))
    comp_indices = make_composite_index(g_vec)
    for I,i_vec in enumerate(comp_indices):
        for J,j_vec in enumerate(comp_indices): 
            for di,si in enumerate(i_vec):
                for dj,sj in enumerate(j_vec):
                    H[I,J] *= mat_list[(di,dj)][si,sj]
    return H 

def weights_product_f(x,g_vec):
    mat_list = aux_functions.vec_to_weights_matrix(x,g_vec)
    n_elem_H = get_num_multi_groups(g_vec)
    H = np.ones((n_elem_H,n_elem_H))
    comp_indices = make_composite_index(g_vec)
    for I,i_vec in enumerate(comp_indices):
        for J,j_vec in enumerate(comp_indices):
            for d,j in enumerate(j_vec):
                H[I,J] *= mat_list[d][I,j]
    return H

def weights_mean_f(x,g_vec):
    mat_list = aux_functions.vec_to_weights_matrix(x,g_vec)
    n_elem_H = get_num_multi_groups(g_vec)
    H = np.zeros((n_elem_H,n_elem_H))
    comp_indices = make_composite_index(g_vec)
    for I,i_vec in enumerate(comp_indices):
        for J,j_vec in enumerate(comp_indices):
            for d,j in enumerate(j_vec):
                H[I,J] += mat_list[d][I,j]/len(g_vec)
    return H

def weights_mean_allsame_f(x,g_vec):
    x_preference = x[:-2]
    mat_list = aux_functions.vec_to_weights_matrix_allsame(x_preference,g_vec)
    n_elem_H = get_num_multi_groups(g_vec)
    H = np.zeros((n_elem_H,n_elem_H))
    comp_indices = make_composite_index(g_vec)
    for I,i_vec in enumerate(comp_indices):
        for J,j_vec in enumerate(comp_indices):
            for d,j in enumerate(j_vec):
                H[I,J] += mat_list[d][I,j]/len(g_vec)
        H[I,J] = aux_functions.logistic_function(H[I,J],x[-1],x[-2])
    return H

def weights_product_allsame_f(x,g_vec):
    x_preference = x[:-2]
    mat_list = aux_functions.vec_to_weights_matrix_allsame(x_preference,g_vec)
    n_elem_H = get_num_multi_groups(g_vec)
    H = np.ones((n_elem_H,n_elem_H))
    comp_indices = make_composite_index(g_vec)
    for I,i_vec in enumerate(comp_indices):
        for J,j_vec in enumerate(comp_indices):
            for d,j in enumerate(j_vec):
                H[I,J] *= mat_list[d][I,j]
        H[I,J] = aux_functions.logistic_function(H[I,J],x[-1],x[-2])
    return H