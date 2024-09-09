import itertools
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import copy

from netin.multidim.infer import inference
from netin.multidim.infer import aux_functions

def MRQAP_1dSimple(
    nodes_list,
    edges_list,
    dimensions_list,
    all_attributes_dict,
    mrqap_iter,
    aggr_fun):

    multidim_groups = aux_functions.build_multidim_groups(dimensions_list,all_attributes_dict)
    ## Compute group sizes
    n0,counts0 = aux_functions.get_n_and_counts(nodes_list,edges_list,dimensions_list)
    ## Sort them according to pre-chosen order
    n, counts = aux_functions.sort_n_and_counts(n0,counts0,multidim_groups)

    g_vec = n.columns.levshape

    H_est_simple, x_est_orig, num_params = inference.estimate_H(n,counts,type_p=aggr_fun,print_convergence=False)

    num_params = len(x_est_orig)
    X_rnd = np.zeros((mrqap_iter,num_params))
    X_rnd_norm = np.zeros((mrqap_iter,num_params))

    ## Initialize node list copy to randomize
    nodes_list_rnd = copy.deepcopy(nodes_list)

    for it in tqdm(range(mrqap_iter),position=0,leave=True):
        
        ## Randomize nodes order
        nodes_list_rnd[:] = nodes_list.sample(frac=1,replace=False).values
        
        ## Count inter-group links
        n_rnd0,counts_rnd0 = aux_functions.get_n_and_counts(nodes_list_rnd,edges_list,dimensions_list)
        ## To ensure consistent ordering
        new_cols = [i for i in multidim_groups if i in n_rnd0.columns] 
        n_rnd = n_rnd0.reindex(columns=new_cols)
        
        new_index = [i for i in multidim_groups if i in counts_rnd0.index]
        new_cols = [i for i in multidim_groups if i in counts_rnd0.columns]
        counts_rnd = counts_rnd0.reindex(index=new_index,columns=new_cols)

        ## Infer preferences
        _, x_rnd, _ = inference.estimate_H(n_rnd,counts_rnd,type_p=aggr_fun,print_convergence=False)
        X_rnd[it,:] = x_rnd
        
        ## Normalize x_rnd
        h_mtrx_rnd = aux_functions.vec_to_mat_list(x_rnd,g_vec)
        h_mtrx_rnd_norm = [np.divide(i.T, np.diag(i)).T for i in h_mtrx_rnd]
        x_rnd_norm = np.array(list(itertools.chain.from_iterable([i.ravel() for i in h_mtrx_rnd_norm])))
        X_rnd_norm[it,:] = x_rnd_norm
        
    X_av = np.mean(X_rnd,axis=0)
    X_std = np.std(X_rnd,axis=0)
    X_norm_av = np.mean(X_rnd_norm,axis=0)
    X_norm_std = np.std(X_rnd_norm,axis=0)

    ## Normalized version of empirical preferences
    h_mtrx_est_orig = aux_functions.vec_to_mat_list(x_est_orig,g_vec)
    h_mtrx_est_orig_norm = [np.divide(i.T, np.diag(i)).T for i in h_mtrx_est_orig]
    x_est_orig_norm = np.array(list(itertools.chain.from_iterable([i.ravel() for i in h_mtrx_est_orig_norm])))

    ## Compute p-value
    X_pval_oneside = np.sum(np.less_equal(X_rnd_norm,x_est_orig_norm),axis=0) / X_rnd.shape[0] ## Proportion of iterations where preference is less than empirical
    assert np.all(X_pval_oneside>=0) and np.all(X_pval_oneside<=1) ## To avoid further mistakes
    X_pval_twoside = copy.deepcopy(X_pval_oneside)
    X_pval_twoside[X_pval_twoside>0.5] = 1-X_pval_twoside[X_pval_twoside>0.5]
    ## Careful, this is not the two-sided, which would be 2 times that value if the distribution is symmetric

    data_and_labels = [
        (X_av,"MRQAP_av_h_"),
        (X_std,"MRQAP_std_h_"),
        (X_norm_av,"MRQAP_av_h_norm_"),
        (X_norm_std,"MRQAP_std_h_norm_"),
        (X_pval_oneside,"MRQAP_pval1s_h_"),
        (X_pval_twoside,"MRQAP_pval2s_h_")
        ]

    ## Extract and save 1D average preferences
    num_dimensions = len(dimensions_list)
    att_counts_rnd0 = [counts_rnd.T.groupby(level=i,sort=False).sum().T.groupby(level=i, sort=False).sum() for i in range(num_dimensions)]
    ## To ensure consistent ordering
    att_counts_rnd = []
    for i, cnts in enumerate(att_counts_rnd0):
        assert set(all_attributes_dict[dimensions_list[i]]).intersection(set(cnts.index))
        assert set(all_attributes_dict[dimensions_list[i]]).intersection(set(cnts.columns))
        new_index = [attr_i for attr_i in all_attributes_dict[dimensions_list[i]] if attr_i in cnts.index]
        new_cols = [attr_i for attr_i in all_attributes_dict[dimensions_list[i]] if attr_i in cnts.columns]
        att_counts_rnd.append( cnts.reindex(index=new_index, columns=new_cols) )

    results_dictionary = {}

    for x_est, name_prefix in data_and_labels:
        
        h_est_mtrx = aux_functions.vec_to_mat_list(x_est,g_vec)
        h_est_simple = []
        for i in range(num_dimensions):
            h_est_simple.append(pd.DataFrame(h_est_mtrx[i],index=att_counts_rnd[i].index,columns=att_counts_rnd[i].columns))

        results_dictionary[name_prefix[:-1]] = h_est_simple

    return results_dictionary