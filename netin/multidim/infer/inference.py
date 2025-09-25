import numpy as np
import pandas as pd
from scipy.stats import binom
from scipy.stats import multinomial
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from netin.multidim.generate.utils import get_num_multi_groups
from netin.multidim.infer import aux_functions
import netin.multidim.infer.aggregation_functions as agg_f

#np.random.seed(42)

''' Likelihood Functions '''

def binomial_loglikelihood(p_est,counts,NN):
    return np.sum(counts*np.log(p_est) + (NN-counts)*np.log(1-p_est))


''' Maximization of likelihood Function '''

def maximize_likelihood(population,link_count,likelihood_function,aggregation_function,x_0, print_convergence=False,**kwargs):
    NN = aux_functions.get_NN(population)
    m = population.columns.levshape[0]**2
    g_vec = population.columns.levshape
    res = minimize(lambda x:-likelihood_function(aggregation_function(x,g_vec),link_count.values,NN.values), x_0, **kwargs)
    if print_convergence: print('Likelihood maximization convergence result: '+res.message)
    p_est = aggregation_function(res.x,g_vec)
    return pd.DataFrame(p_est,index=link_count.index,columns=link_count.columns), res.x


''' Function to get the estimate for H '''

def estimate_H(population,link_count,type_p='multidimensional',opt_method = 'L-BFGS-B',opt_options = {'ftol':1e-16,'gtol':1e-16,'maxfun':100000}, print_convergence=False):

    if np.prod(link_count.shape)==0: raise ValueError("The filtered dataset is empty.")
        
    L = link_count.sum().sum()
    n_attributes = population.columns.nlevels
    g_vec = population.columns.levshape
    NN = aux_functions.get_NN(population)
    p = link_count.div(NN)
    num_nans = p.isna().sum().sum()
    p.fillna(0,inplace=True) # It doesn't influence the likelihood

    att_pop = [population.groupby(level=i, axis=1, sort=False).sum() for i in range(n_attributes)]
    att_counts = [link_count.groupby(level=i, axis=1, sort=False).sum().groupby(level=i, axis=0, sort=False).sum() for i in range(n_attributes)]
    att_p = [att_counts[i].div(aux_functions.get_NN(att_pop[i])) for i in range(n_attributes)]
    for t in att_p: t.fillna(0,inplace=True) # It doesn't influence the likelihood

    m = sum(np.square(g_vec))
    x_0 = np.random.rand(m)
    opt_bounds = [(1e-10,1-1e-10) for i in range(m)]
    
    p_est = None
    params = None
    num_param = None

    if type_p=='multidimensional':
        p_est = p
        params = p.values.reshape(np.prod(p.shape))
        num_params = np.prod(p.shape)-num_nans
    
    elif type_p[:10]=='dimension_':
        dim = int(type_p[10])-1
        J_list = [np.ones((i,i)) for i in g_vec]
        mat_list = list(J_list[:dim]) + [att_p[dim]] + list(J_list[dim+1:])
        p_est = pd.DataFrame(aux_functions.nested_f(mat_list,np.kron),index=p.index,columns=p.columns)
        params = att_p[dim].values.reshape(np.prod(att_p[dim].shape))
        num_params = np.prod(att_p[dim].shape)
    
    elif type_p=='and':
        ## The initial guess x_0 for the param estimates (1d preference) is the one-dimensional density (same for other functions)
        x_0 = np.concatenate([att_p[i].values.reshape(g_vec[i]**2) for i in range(n_attributes)])
        p_est,params = maximize_likelihood(population,link_count,binomial_loglikelihood,agg_f.product_f,x_0,bounds=opt_bounds, method=opt_method, options=opt_options, print_convergence=print_convergence)
        num_params = np.sum(np.square(g_vec))
    
    elif type_p=='or':
        x_0 = np.concatenate([att_p[i].values.reshape(g_vec[i]**2) for i in range(n_attributes)])
        p_est,params = maximize_likelihood(population,link_count,binomial_loglikelihood,agg_f.at_least_one_f,x_0,bounds=opt_bounds, method=opt_method, options=opt_options, print_convergence=print_convergence)
        num_params = np.sum(np.square(g_vec))
    
    elif type_p=='mean':
        x_0 = np.concatenate([att_p[i].values.reshape(g_vec[i]**2) for i in range(n_attributes)])
        p_est,params = maximize_likelihood(population,link_count,binomial_loglikelihood,agg_f.mean_f,x_0,bounds=opt_bounds, method=opt_method, options=opt_options, print_convergence=print_convergence)
        num_params = np.sum(np.square(g_vec))
    
    elif type_p=='and_trivial':
        x = np.concatenate([att_p[i].values.reshape(g_vec[i]**2) for i in range(n_attributes)])
        p_est = pd.DataFrame(agg_f.product_f(x,g_vec),index=p.index,columns=p.columns)
        s = (p_est*NN).sum().sum()
        p_est = p_est*(L/s)
        params = x
        num_params = np.sum(np.square(g_vec))
    
    elif type_p=='or_trivial':
        x = np.concatenate([att_p[i].values.reshape(g_vec[i]**2) for i in range(n_attributes)])
        p_est = pd.DataFrame(agg_f.at_least_one_f(x,g_vec),index=p.index,columns=p.columns)
        s = (p_est*NN).sum().sum()
        p_est = p_est*(L/s)
        params = x
        num_params = np.sum(np.square(g_vec))
        
    elif type_p=='mean_trivial':
        x = np.concatenate([att_p[i].values.reshape(g_vec[i]**2) for i in range(n_attributes)])
        p_est = pd.DataFrame(agg_f.mean_f(x,g_vec),index=p.index,columns=p.columns)
        s = (p_est*NN).sum().sum()
        p_est = p_est*(L/s)
        params = x
        num_params = np.sum(np.square(g_vec))

    elif type_p[:8]=='and_dim_':
        minus_ind = type_p.rfind('-')
        dims = [int(type_p[8:minus_ind])-1,int(type_p[minus_ind+1:])-1]
        restricted_pop = population.groupby(level=dims, axis=1, sort=False).sum()
        restricted_counts = link_count.groupby(level=dims, axis=1, sort=False).sum().groupby(level=dims, axis=0, sort=False).sum()
        restricted_p = restricted_counts.div(aux_functions.get_NN(restricted_pop))
        restricted_p.fillna(0,inplace=True) # It doesn't influence the likelihood
        x_0 = restricted_p.values.reshape(np.prod(restricted_counts.shape))
        opt_bounds = [(1e-10,1-1e-10) for i in range(len(x_0))]
        _,x = maximize_likelihood(restricted_pop,restricted_counts,binomial_loglikelihood,agg_f.product_f,x_0,bounds=opt_bounds, method=opt_method, options=opt_options, print_convergence=print_convergence)
        x_to_mat_list = aux_functions.vec_to_mat_list(x,[g_vec[i] for i in dims])
        mat_list = [np.ones((g_vec[i],g_vec[i])) if (i not in dims) else x_to_mat_list[dims.index(i)] for i in range(len(g_vec))]
        p_est = pd.DataFrame(aux_functions.nested_f(mat_list,np.kron),index=p.index,columns=p.columns)
        params = x
        num_params = len(x)

    elif type_p=='product_cross_dimensional':
        num_param = np.sum(g_vec)**2
        opt_bounds = [(1e-10,1-1e-10) for i in range(num_param)]
        x_0 = np.random.rand(num_param)
        p_est, params = maximize_likelihood(population,link_count,binomial_loglikelihood,agg_f.cross_dimensional_product_f,x_0,bounds=opt_bounds, method=opt_method, options=opt_options, print_convergence=print_convergence)
        num_params = len(params)

    elif type_p=='product_weights':
        num_multi_groups = get_num_multi_groups(g_vec)
        num_param = num_multi_groups*np.sum(g_vec)
        opt_bounds = [(1e-10,1-1e-10) for i in range(num_param)]
        x_0 = np.random.rand(num_param)
        p_est, params = maximize_likelihood(population,link_count,binomial_loglikelihood,agg_f.weights_product_f,x_0,bounds=opt_bounds, method=opt_method, options=opt_options, print_convergence=print_convergence)
        num_params = len(params)

    elif type_p=='mean_weights':
        num_multi_groups = get_num_multi_groups(g_vec)
        num_param = num_multi_groups*np.sum(g_vec)
        opt_bounds = [(1e-10,1-1e-10) for i in range(num_param)]
        x_0 = np.random.rand(num_param)
        p_est, params = maximize_likelihood(population,link_count,binomial_loglikelihood,agg_f.weights_mean_f,x_0,bounds=opt_bounds, method=opt_method, options=opt_options, print_convergence=print_convergence)
        num_params = len(params)

    elif type_p=='weights_product_allsame':
        num_param = len(g_vec)*2+2
        opt_bounds = [(1e-10,1-1e-10) for i in range(num_param-1)] + [(1e-10,np.inf)]
        x_0 = np.random.rand(num_param)
        p_est, params = maximize_likelihood(population,link_count,binomial_loglikelihood,agg_f.weights_product_allsame_f,x_0,bounds=opt_bounds, method=opt_method, options=opt_options, print_convergence=print_convergence)
        num_params = len(params)

    elif type_p=='mean_weights_allsame':
        num_param = len(g_vec)*2+2
        opt_bounds = [(1e-10,1-1e-10) for i in range(num_param-1)] + [(1e-10,np.inf)]
        x_0 = np.random.rand(num_param)
        p_est, params = maximize_likelihood(population,link_count,binomial_loglikelihood,agg_f.weights_mean_allsame_f,x_0,bounds=opt_bounds, method=opt_method, options=opt_options, print_convergence=print_convergence)
        num_params = len(params)

    else:
        raise ValueError(f"The preference type {type_p} does not correspond to any implemented preference.")
        
    return p_est, params, num_params


''' Function to compute the likelihood, AIC and BIC '''

def compute_likel(population,link_counts,p,type_m='binom',k=1,print_values=True,name=''):

    NN = aux_functions.get_NN(population)
    L = link_counts.sum().sum()
    g_vec = population.columns.levshape
    dim = int(np.prod(g_vec))**2
    log_likel = 0
    
    if type_m=='binom':
        for col in link_counts.columns:
            for row in link_counts[col].index:
                log_bin_p = np.log(binom.pmf(link_counts.loc[row,col], NN.loc[row,col], p.loc[row,col]))
                log_likel = log_likel + log_bin_p

    if type_m=='multinom':
        log_likel = np.log(multinomial.pmf(link_counts.values.reshape(dim), n=L, p=(p.div(p.sum().sum())).values.reshape(dim)))

    AIC_val = 2*k - 2*log_likel
    BIC_val = k*np.log(dim) - 2*log_likel

    if print_values:
        print(f'Likelihood {name}: {log_likel}')
        print(f'AIC {name}: {AIC_val}')
        print(f'BIC {name}: {BIC_val}')

    return log_likel, AIC_val, BIC_val,k


''' Wrapper to calculate AIC and BIC for all aggregation functions '''

def create_table(population, link_count, types_p=None, print_convergence=False):
    g_vec = population.columns.levshape
    data = []
    if types_p is None: 
        types_p = ['multidimensional','and','or','mean']+[f'dimension_{i+1}' for i in range(len(g_vec))] + [f'and_dim_{i+1}-{j+1}' for i in range(len(g_vec)) for j in [x for x in list(range(len(g_vec))) if x>i]]
    for type_p in types_p:
        p_est, _, num_params = estimate_H(population,link_count,type_p=type_p,print_convergence=print_convergence)
        data.append([type_p]+list(compute_likel(population,link_count,p_est,k=num_params,print_values=False)))
    results = pd.DataFrame(data,columns=['Type of model','Likelihood','AIC','BIC','N parameters'])
    return results


''' Function to plot the true density against the estimated one '''

def plot_similarity(counts,n,p_est):
    NN = aux_functions.get_NN(n)
    p = counts.div(NN)
    dim = p.shape[0]*p.shape[1]

    fig = plt.figure(figsize=(7, 5))
    plt.scatter(p.values.reshape((dim,1)),p_est.values.reshape((dim,1)))
    m = max(p.max(axis=None),p_est.max(axis=None))
    plt.plot((0,m),(0,m),'red')
    plt.xlabel('True H')
    plt.ylabel('Estimated H')
    plt.show()



