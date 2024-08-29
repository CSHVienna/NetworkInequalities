import numpy as np
import networkx as nx
from tqdm.auto import tqdm

from netin.multidim.generate.validators import is_pop_frac_consistent
from netin.multidim.generate.utils import make_composite_index
from netin.multidim.generate.utils import get_num_multi_groups
from netin.multidim.generate.utils import comp_index_to_integer
from netin.multidim.theory.multidimensional_preferences import composite_H_ij_hierarchy

def multidimensional_network_fix_av_degree(
    h_mtrx_lst,
    comp_pop_frac_tnsr,
    kind,
    directed=True,
    pop_fracs_lst = None,
    m=3,
    N=1000,
    v = 1,
    alpha = None,
    p_d = None
    ):
    """Generate a multidimensional graph with fixed average degree

    Creates a network with N disconnected nodes and then conntects them using
    a stochastic block-model like mechanism where the connection  probabilities
    are computed from the one-dimensional preferences (h_mtrx_lst)
    according to the type of aggregation function (kind).

    In this version, N*m links are created among the N nodes.
        
    Parameters
    ----------
    h_mtrx_lst : np.ndarray
        One-dimensional latent preferences as list of matrices

    comp_pop_frac_tnsr : np.ndarray
        Population fractions tensor

    kind : str
        Type of aggregation function

    directed : bool, optional
        Directed or undirected Graph

    pop_fracs_lst : Iterable, optional
        Population marginal distributions as list of np.array

    m : int, optional
        Average degree (the default is 3)

    N : int, optional
        Number of nodes (the default is 1000)

    v : int, optional
        verbosity

    alpha : float, optional
        Parameter for aggregation functions min and max.

    p_d : Iterable, optional
        Parameter for aggregation function one (MEAN in the paper). 
        Weight of each dimension for the weighted average.

    Returns
    -------
    nx.Graph
        Resulting network
    """


    h_mtrx_lst = np.array(h_mtrx_lst)

    ## Assert that every parameter is within appropriate ranges
    am_preliminary_checks(
        h_mtrx_lst,
        comp_pop_frac_tnsr,
        pop_fracs_lst=pop_fracs_lst)

    ## Compute number of dimensions
    D = len(h_mtrx_lst)
    assert D == comp_pop_frac_tnsr.ndim    

    ## Build random population of nodes alla Centola
    G = build_social_structure(N,comp_pop_frac_tnsr,directed)

    ## Build interaction function (faster than an if-switch inside 
    ## the inner for loop)
    if kind == "any":
        interaction = lambda h_vec,param: interaction_any(h_vec,param)
        param = None
    elif kind == "all":
        interaction = lambda h_vec,param: interaction_all(h_vec,param) 
        param = None
    elif kind == "one":
        assert len(p_d) == D
        interaction = lambda h_vec,param: interaction_one(h_vec,param)
        param = p_d
    elif kind == "max":
        interaction = lambda h_vec,param: interaction_max(h_vec,param)
        param = alpha
    elif kind == "min":
        interaction = lambda h_vec,param: interaction_min(h_vec,param)
        param = alpha
    elif kind == "hierarchy":
        interaction = lambda h_vec,param: interaction_hierarchy(h_vec,param)
    else:
        raise ValueError(f"Interaction kind {kind} invalid.")

    ## Iterate
    h_vec = np.zeros(D)
    n_lnks = 0

    ####
    ## DEBUG: Potential links counter
    # ptn_lnk_dct = {}
    # succ_lnk_dct = {}
    ####

    while n_lnks < N*m:
    # for i in range(N*m):
        if v==1 and n_lnks%1000 == 0:
            print (n_lnks)
        ## Random node 1 and 2
        n, target = np.random.randint(N,size=2)
        if n == target:
            continue
        ## Check if link exists
        if G.has_edge(n,target):
            continue
        ## Compute homophily
        orig_idx = G.nodes[n]["attr"]
        target_idx = G.nodes[target]["attr"]

        ####
        ## DEBUG
        # try:
        #     ptn_lnk_dct[(orig_idx,target_idx)] += 1
        # except KeyError:
        #     ptn_lnk_dct[(orig_idx,target_idx)] = 1
        ####

        if kind == "hierarchy":
            h_vec = h_mtrx_lst
            param = (orig_idx,target_idx)
        else:
            for d in range(D):
                h_vec[d] = h_mtrx_lst[
                        d,
                        orig_idx[d],
                        target_idx[d]
                        ]

        ####
        ## DEBUG
        # print (orig_idx, target_idx, h_vec)
        ####

        ## Check if the tie is made
        successful_tie = interaction(h_vec,param)

        ## Create links
        if successful_tie:
            G.add_edge(n,target)
            n_lnks+=1

            ####
            ## DEBUG
            # try:
            #     succ_lnk_dct[(orig_idx,target_idx)] += 1
            # except KeyError:
            #     succ_lnk_dct[(orig_idx,target_idx)] = 1
            ####

    ####
    ## DEBUG
    # print ("Potential links",ptn_lnk_dct,succ_lnk_dct)
    ####

    return G

def multidimensional_network(
    h_mtrx_lst,
    comp_pop_frac_tnsr,
    kind,
    directed=True,
    pop_fracs_lst = None,
    # m=3,
    N=1000,
    v = 0,
    get_aggr_res = False,
    alpha = None,
    p_d = None,
    ):
    """Generate a multidimensional graph

    Creates a network with N disconnected nodes and then conntects them using
    a stochastic block-model like mechanism where the connection  probabilities
    are computed from the one-dimensional preferences (h_mtrx_lst)
    according to the type of aggregation function (kind).
        
    Parameters
    ----------
    h_mtrx_lst : np.ndarray
        One-dimensional latent preferences as list of matrices

    comp_pop_frac_tnsr : np.ndarray
        Population fractions tensor

    kind : str
        Type of aggregation function

    directed : bool, optional
        Directed or undirected Graph

    pop_fracs_lst : Iterable, optional
        Population marginal distributions as list of np.array

    # m : int, optional
        Average degree [not used] (the default is 3)

    N : int, optional
        Number of nodes (the default is 1000)

    v : int, optional
        verbosity (the default is 0, which suppresses all messages)

    get_aggr_res : bool, optional
        To get aggregated results as multidimensional mixing matrix instead
        of a network. Saves a lot of computation time and memory.

    alpha : float, optional
        Parameter for aggregation functions min and max.

    p_d : Iterable, optional
        Parameter for aggregation function one (MEAN in the paper). 
        Weight of each dimension for the weighted average.

    Returns
    -------
    nx.Graph
    	If get_aggr_res=False

	(np.ndarray, np.ndarray)
		if get_aggr_res=True
    """

    h_mtrx_lst = np.array(h_mtrx_lst)

    ## Assert that every parameter is within appropriate ranges
    am_preliminary_checks(
        h_mtrx_lst,
        comp_pop_frac_tnsr,
        pop_fracs_lst=pop_fracs_lst)

    ## Compute number of dimensions
    D = len(h_mtrx_lst)
    assert D == comp_pop_frac_tnsr.ndim

    ## Compute number of groups per dimension (v_d)
    g_vec = [len(h) for h in h_mtrx_lst]    

    ## Build random population of nodes alla Centola
    G = build_social_structure(N,comp_pop_frac_tnsr,directed)

    ## Build interaction function (faster than an if-switch inside 
    ## the inner for loop)
    if kind == "any":
        interaction = lambda h_vec,param: interaction_any(h_vec,param)
        param = None
    elif kind == "all":
        interaction = lambda h_vec,param: interaction_all(h_vec,param) 
        param = None
    elif kind == "one":
        assert len(p_d) == D
        interaction = lambda h_vec,param: interaction_one(h_vec,param)
        param = p_d
    elif kind == "max":
        interaction = lambda h_vec,param: interaction_max(h_vec,param)
        param = alpha
    elif kind == "min":
        interaction = lambda h_vec,param: interaction_min(h_vec,param)
        param = alpha
    elif kind == "hierarchy":
        interaction = lambda h_vec,param: interaction_hierarchy(h_vec,param)
    else:
        raise ValueError(f"Interaction kind {kind} invalid.")

    ################
    ## PRELIMINARY - IT DOESN'T WORK FOR HIGH DIMENSIONS (it depends on preference
    ## aggregation function)!!
    # rho_tgt = m/N
    # rho = 0
    # for i in range(h_mtrx_lst[0].shape[0]):
    #     for j in range(h_mtrx_lst[0].shape[0]):
    #         rho += h_mtrx_lst[0][i,j] * comp_pop_frac_tnsr[i] * comp_pop_frac_tnsr[j]

    # rho_factor = rho_tgt / rho
    # print ("Target density / Rho factor", rho_tgt, rho_factor)
    ################

    ## Init counts matrix if asked
    if get_aggr_res:
        num_groups = get_num_multi_groups(g_vec)
        M_cnts_dir = np.zeros((num_groups,num_groups))
        pop_cnts = np.zeros(num_groups)
        for n in range(N):
            I = comp_index_to_integer(G.nodes[n]["attr"],g_vec)
            pop_cnts[I] += 1

    ## Initialize the homophily interaction vector
    h_vec = np.zeros(D)
    # n_lnks = 0

    for n in tqdm(range(N),disable=1-v):
        for target in range(N):

            ################
            ## Randomly pick link to achieve target density
            # if np.random.random() > rho_factor:
            #     continue
            ################

            ## Compute homophily
            orig_idx = G.nodes[n]["attr"]
            target_idx = G.nodes[target]["attr"]

            ## We allow self loops. Multi-edges are impossible
            ## because each potential link is only attempted once.

            if kind == "hierarchy":
                h_vec = h_mtrx_lst
                param = (orig_idx,target_idx)
            else:
                for d in range(D):
                    h_vec[d] = h_mtrx_lst[
                            d,
                            orig_idx[d],
                            target_idx[d]
                            ]

            ## Check if the tie is made
            successful_tie = interaction(h_vec,param)

            ## Create links
            if successful_tie:
                if get_aggr_res:
                    I = comp_index_to_integer(orig_idx,g_vec)
                    J = comp_index_to_integer(target_idx,g_vec)
                    M_cnts_dir[I,J] += 1
                else:
                    G.add_edge(n,target)
                # n_lnks+=1

    # print ("Final density", n_lnks / N**2.0)
    
    if get_aggr_res:
        return M_cnts_dir, pop_cnts
    else:
        return G

##############################################################################
## Preliminary checks
##############################################################################

def am_preliminary_checks(*args,**kwargs):
    h_mtrx_lst, comp_pop_frac_tnsr = args
    pop_fracs_lst = kwargs["pop_fracs_lst"]

    h_mtrx_lst = np.array(h_mtrx_lst)
    ## Check range of parameters
    assert np.all(h_mtrx_lst<=1)
    assert np.all(h_mtrx_lst>=0)

    if pop_fracs_lst is not None:
        assert is_pop_frac_consistent(pop_fracs_lst,comp_pop_frac_tnsr)
    else:
        assert np.abs(np.sum(comp_pop_frac_tnsr) - 1.0) < 1e-13

    ## Check that simple homophilies are row-normalized
#     for h_mtrx in h_mtrx_lst:
#         g = len(h_mtrx)
#         ref = np.ones(g)
#         assert np.sum(np.abs(np.sum(h_mtrx,axis=1) - ref)) < 1e-13

##############################################################################
## Initialize empty network
#############################################################################

def build_probs_pop(comp_pop_frac_tnsr):
    g_vec = comp_pop_frac_tnsr.shape
    X = make_composite_index(g_vec)
    p = []
    for i_vec in X:
        p.append(comp_pop_frac_tnsr[i_vec])
    assert np.abs(np.sum(p) - 1.0) < 1e-13
    return X, p

def build_social_structure(N,comp_pop_frac_tnsr,directed):
    ## Build probability distribution of composite populations
    memberships, probs = build_probs_pop(comp_pop_frac_tnsr)
    memberships_idx = list(range(len(memberships)))
    ## Assign a membership to each node
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    for n in range(N):
        node_type_idx = np.random.choice(memberships_idx,p=probs)
        node_type = memberships[node_type_idx]
        G.add_node(n,attr=node_type)
    return G

##############################################################################
## Kinds of interaction for the agent model
##############################################################################

def interaction_all(h_vec,nothing):
    # assert np.all(h_vec > 0) ## Otherwise we would end up in inifinite loop
    # if np.sum(np.random.random(size=len(h_vec)) <= h_vec) == len(h_vec):
    #     return True
    if np.all(np.random.random(size=len(h_vec)) <= h_vec):
        return True
    return False

def interaction_any(h_vec,nothing):
    for hd in h_vec:
        if np.random.random() <= hd:
            return True
    return False

def interaction_one(h_vec,p_d):
    hd = np.random.choice(h_vec,p=p_d)
    if np.random.random() <= hd:
        return True
    return False

def interaction_max(h_vec,alpha):
    if np.random.random() <= alpha:
        hd = max(h_vec)
    else:
        hd = np.random.choice(h_vec)

    if np.random.random() <= hd:
        return True
    return False

def interaction_min(h_vec,alpha):
    if np.random.random() <= alpha:
        hd = min(h_vec)
    else:
        hd = np.random.choice(h_vec)

    if np.random.random() <= hd:
        return True
    return False

def interaction_hierarchy(h_mtrx_lst,param):
    i_vec, j_vec = param
    h = composite_H_ij_hierarchy(i_vec,j_vec,h_mtrx_lst)
    if np.random.random() <= h:
        return True
    return False