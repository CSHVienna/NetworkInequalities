import numpy as np
import pandas as pd
from itertools import product
from scipy import special

from netin.multidim.generate.utils import get_num_multi_groups
from netin.multidim.generate.utils import make_composite_index

def build_multidim_groups(
    dimensions_list,
    all_attributes_dict):
    
    num_dimensions = len(dimensions_list)
    multidim_groups = list(product(*[all_attributes_dict[d] for d in dimensions_list]))

    return multidim_groups

def get_n_and_counts(nodes_df,edges_df,attributes,index=None):
    if index is not None:
        nodes_df = nodes_df.set_index(index)
    for x in attributes:
        edges_df['source '+x] =  edges_df['source'].map(nodes_df[x])
        edges_df['target '+x] =  edges_df['target'].map(nodes_df[x])
    pop = pd.DataFrame(pd.crosstab([nodes_df[x] for x in attributes[:-1]], nodes_df[attributes[-1]]).stack(),columns=['N']).transpose()
    indexes = pd.MultiIndex.from_tuples(list(product(*[list(x) for x in pop.columns.levels])), names=pop.columns.names)
    population = pd.DataFrame(0,index=['N'],columns=indexes)
    population.update(pop)
    link_count = pd.DataFrame(0,index=population.columns, columns=population.columns)
    link_count.update(pd.crosstab([edges_df["source "+x] for x in attributes], [edges_df["target "+x] for x in attributes]))
    return population,link_count

def sort_n_and_counts(n0, counts0, multidim_groups):
    """
    
    Sort the columns and indices in the population (n) and inter-group link
    counts (counts) dataframes according to the 
    
    Parameters
    ----------
    n0 : pandas.DataFrame
        Original population dataframe.

    counts0 : pandas.DataFrame
        Original inter-group counts dataframe.

    multidim_groups : iterable
        List (or similar) of tuples, wher each tuple is a multidimensional
        group.
    
    Returns
    -------
    (pandas.DataFrame, pandas.DataFrame)
        New n and counts dataframes sorted according to multidim_groups.
    """
    ## To ensure CONSISTENT ORDERING
    ## First extract only the columns that exist in the dataframe, otherwise we 
    ## run into problems because, even if we drop NaNs, the resulting dataframe, 
    ## internally, thinks that there are more nonempty columns than there really are
    new_cols = [i for i in multidim_groups if i in n0.columns] 
    n = n0.reindex(columns=new_cols)#.dropna(how="all",axis=1) ## If we use this instead of "new_cols" we end up with empty columns that change n.columns.levshape which is used later for many computations (g_vec=n.columns.levshape)

    new_index = [i for i in multidim_groups if i in counts0.index]
    new_cols = [i for i in multidim_groups if i in counts0.columns]
    counts = counts0.reindex(index=new_index,columns=new_cols       ## Change order of columns and rows
                            )#.dropna(how="all",axis=0).dropna(how="all",axis=1) ## Remove empty rows and columns

    return n, counts

def one_dim_groups_counts(
    n,
    counts,
    all_attributes_dict,
    dimensions_list):

    num_dimensions = len(dimensions_list)
    ## Compute 1D group sizes and inter-group links
    att_pop0 = [n.T.groupby(level=i, sort=False).sum() for i in range(num_dimensions)]
    att_counts0 = [counts.T.groupby(level=i,sort=False).sum().T.groupby(level=i, sort=False).sum() for i in range(num_dimensions)]

    ## To ensure consistent ordering
    att_counts = []
    for i, cnts in enumerate(att_counts0):
        assert set(all_attributes_dict[dimensions_list[i]]).intersection(set(cnts.index))
        assert set(all_attributes_dict[dimensions_list[i]]).intersection(set(cnts.columns))
        new_index = [attr_i for attr_i in all_attributes_dict[dimensions_list[i]] if attr_i in cnts.index]
        new_cols = [attr_i for attr_i in all_attributes_dict[dimensions_list[i]] if attr_i in cnts.columns]
        att_counts.append( cnts.reindex(index=new_index, columns=new_cols) )

    att_pop = []
    for i, cnts in enumerate(att_pop0):
        assert set(all_attributes_dict[dimensions_list[i]]).intersection(set(cnts.index))
        new_index = [attr_i for attr_i in all_attributes_dict[dimensions_list[i]] if attr_i in cnts.index]
        att_pop.append( cnts.reindex(index=new_index) )

    return att_pop, att_counts

def get_NN(n):
    return n.transpose(copy=True).dot(n) - pd.DataFrame(np.diag(n.values.reshape((n.shape[1]))),index=n.columns,columns=n.columns)

def nested_f(x,fun):
    if len(x)==2: return fun(x[0],x[1])
    else: return fun(x[0],nested_f(x[1:],fun))   

def vec_to_mat_list(x,g_vec):
    indeces = [0]+list(np.cumsum(np.square(g_vec)))
    return [x[indeces[i]:indeces[i+1]].reshape((g_vec[i],g_vec[i])) for i in range(len(g_vec))]

def vec_to_mat_dict_cross_one_dimensional(x,g_vec):
    mat_dict = {}
    from_index = 0
    for di,gi in enumerate(g_vec):
        for dj,gj in enumerate(g_vec):
            until_index = from_index + gi*gj
            mat_dict[(di,dj)] = x[from_index:until_index].reshape((gi,gj))
            from_index = until_index
    return mat_dict

def vec_to_weights_matrix(x,g_vec):
    num_multi_groups = get_num_multi_groups(g_vec)
    preference_matrix_lst = [np.zeros((num_multi_groups,gi)) for gi in g_vec]
    from_index = 0
    for d,gi in enumerate(g_vec):
        for I in range(num_multi_groups):
            until_index = from_index + gi
            preference_matrix_lst[d][I,:] = x[from_index:until_index]
            from_index = until_index
    return preference_matrix_lst

def vec_to_weights_matrix_allsame(x,g_vec):
    assert len(x) == 2*len(g_vec) ## in and out-group in each dimension
    in_out_preferences = [x[d*2:(d+1)*2] for d in range(len(g_vec))]
    comp_indices = make_composite_index(g_vec)
    num_multi_groups = len(comp_indices)
    preference_matrix_lst = [np.zeros((num_multi_groups,gi)) for gi in g_vec]
    for I, i_vec in enumerate(comp_indices):
        for d, gi in enumerate(g_vec):
            for si in range(gi):
                if si == i_vec[d]:
                    preference_matrix_lst[d][I,si] = in_out_preferences[d][0]
                else:
                    preference_matrix_lst[d][I,si] = in_out_preferences[d][1]
    return preference_matrix_lst

def logistic_function(x,a,b):
    return 1/(1+np.exp(-a*(x-b)))

def shuffle_df(nodes_list):
    nodes_list[:] = nodes_list.sample(frac=1).values
    return nodes_list

def get_shuffled_ps(nodes_list,edges_list,attributes,N_SHUFFLES,index=None):
    nodes_list = nodes_list.copy()
    nodes_list = nodes_list[attributes]
    n,counts = get_n_and_counts(nodes_list,edges_list,attributes)
    shuffled_ps = np.zeros((N_SHUFFLES,*counts.shape))
    for i in range(N_SHUFFLES):
        nodes_list = shuffle_df(nodes_list)
        n,counts = get_n_and_counts(nodes_list,edges_list,attributes)
        if index is not None:
            n = n.reindex(columns=index)
            counts = counts.reindex(columns=index,index=index)
        NN = get_NN(n)
        p = counts.div(NN)
        pref = p.div(p.sum(axis=1),axis=0)
        shuffled_ps[i,:] = pref.values
    return shuffled_ps

def generate_ps(p,NN,N_ITER,pref=False):
    SHAPE = p.shape
    inferred_p = np.zeros((N_ITER,*SHAPE))
    for i in range(N_ITER):
        inferred_p[i,:] = (np.random.binomial(NN,p)/NN.values)
    if pref: 
        while (np.sum(inferred_p,axis=2)==0).any():
            ind = np.where((np.sum(inferred_p,axis=2)==0).any(axis=1))
            for i in ind:
                inferred_p[i,:] = (np.random.binomial(NN,p)/NN.values)
        inferred_p = inferred_p/np.sum(inferred_p,axis=2).reshape((N_ITER,SHAPE[0],1))
    return inferred_p

## To get the number of free parameters of some of the models
def product_mean_free_params(g_vec):
    g_vec = np.array(g_vec)
    D = len(g_vec)
    return np.sum(g_vec**2) - special.comb(D,2,exact=True)

def full_1d_free_params(g_vec):
    g_vec = np.array(g_vec)
    D = len(g_vec)
    return np.sum(g_vec)**2 - special.comb(D**2,2,exact=True)

def multi_1d_free_params(g_vec):
    g_vec = np.array(g_vec)
    D = len(g_vec)
    return np.prod(g_vec)*np.sum(g_vec) - special.comb(D,2,exact=True)
