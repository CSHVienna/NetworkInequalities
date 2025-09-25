import pandas as pd

from netin.multidim.infer import aux_functions
from netin.multidim.infer import inference

def infer_latent_preferences_1dSimple(
    nodes_list,
    edges_list,
    dimensions_list, 
    all_attributes_dict,
    type_p = "and",
    MRQAP = False):
    
    multidim_groups = aux_functions.build_multidim_groups(dimensions_list,all_attributes_dict)
    ## Compute group sizes
    n0,counts0 = aux_functions.get_n_and_counts(nodes_list,edges_list,dimensions_list)
    ## Sort them according to pre-chosen order
    n, counts = aux_functions.sort_n_and_counts(n0,counts0,multidim_groups)
    ## Compute one-dimensional group sizes and inter-group number of ties
    att_pop, att_counts = aux_functions.one_dim_groups_counts(n,counts,all_attributes_dict,dimensions_list)

    ## One-dimensional model
    H_est_simple, x_est, num_params = inference.estimate_H(n,counts,type_p="and",print_convergence=True)
    g_vec = n.columns.levshape
    num_free_params = aux_functions.product_mean_free_params(g_vec)
    likel_simple = inference.compute_likel(n,counts,H_est_simple,k=num_free_params,print_values=False)
    
    ## Extract and save 1D preferences
    h_est_mtrx = aux_functions.vec_to_mat_list(x_est,g_vec)
    h_est_simple = []
    h_est_norm = []
    num_dimensions = len(dimensions_list)
    for i in range(num_dimensions):
        h_est_simple.append(pd.DataFrame(h_est_mtrx[i],index=att_counts[i].index,columns=att_counts[i].columns))
        h_est_norm.append(h_est_simple[i].div(h_est_simple[i].to_numpy().diagonal(), axis=0))

    infer_results_dct = {
        "multidimensional_population":n,
        "multidimensional_links":counts,
        "one_dimensional_population":att_pop,
        "one_dimensional_links":att_counts,
        "H_multidimensional_preferences":H_est_simple,
        "Likelihood":likel_simple[0],
        "AIC":likel_simple[1],
        "BIC":likel_simple[2],
        "h_inferred_latent_preferences":h_est_simple,
        "h_normalized_inferred_latent_preferences":h_est_norm
        }

    return infer_results_dct

def infer_latent_preferences_Multidimensional(
    nodes_list,
    edges_list,
    dimensions_list, 
    all_attributes_dict):
    
    multidim_groups = aux_functions.build_multidim_groups(dimensions_list,all_attributes_dict)
    ## Compute group sizes
    n0,counts0 = aux_functions.get_n_and_counts(nodes_list,edges_list,dimensions_list)
    ## Sort them according to pre-chosen order
    n, counts = aux_functions.sort_n_and_counts(n0,counts0,multidim_groups)
    ## Compute one-dimensional group sizes and inter-group number of ties
    att_pop, att_counts = aux_functions.one_dim_groups_counts(n,counts,all_attributes_dict,dimensions_list)

    ## One-dimensional model
    H_est_multi, x_est, num_params = inference.estimate_H(n,counts,type_p="multidimensional",print_convergence=True)
    g_vec = n.columns.levshape
    likel_multi = inference.compute_likel(n,counts,H_est_multi,k=num_params,print_values=False)

    infer_results_dct = {
        "multidimensional_population":n,
        "multidimensional_links":counts,
        "one_dimensional_population":att_pop,
        "one_dimensional_links":att_counts,
        "H_multidimensional_preferences":H_est_multi,
        "Likelihood":likel_multi[0],
        "AIC":likel_multi[1],
        "BIC":likel_multi[2]
        }

    return infer_results_dct