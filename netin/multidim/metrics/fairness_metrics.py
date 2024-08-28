from scipy import stats
import numpy as np
from scipy.stats import skellam
import importlib

from netin.multidim.generate.utils import make_composite_index
from netin.multidim.generate.utils import comp_index_to_integer

def theil_index_groups(x_lst):
	"""
	Compute the theil index and its decomposition for N groups.

	Paramteres
	----------
	x_lst: list
		List of lists, where each list contains the wealth of the individuals 
		of a group.

	Returns
	-------
	result: tuple

		result[0]: float
			Total Theil

		result[1]: float
			Total between-group Theil

		result[2]: numpy.ndarray
			Within-group Theil for each group, weighted by their group size and
			wealth.

		result[3]: numpy.ndarray
			Within-group Theil for each group.

	"""
	if len(x_lst) <= 1:
		raise Exception("Function theil_index_groups requires at least 2 groups.")
	## Mask to ignore empty lists if there are any
	msk = np.array([True if len(xi)>0 else False for xi in x_lst])
	## Compute number of groups
	Nk = len(x_lst)
	## Per-group mean
	mu_k = np.array([np.mean(i) for i in x_lst])
	# print (mu_k)
	## Population fractions
	f_k = np.array([len(i) for i in x_lst])
	f_k = f_k / np.sum(f_k)
	# print (f_k)
	## Overall mean
	mu = np.nansum(f_k*mu_k) ## nansum to deal with empty lists
	# print (mu, np.mean(list(x_lst[0])+list(x_lst[1])))
	## Within-group inequalities
	T_k = np.array([theil_index(i) for i in x_lst])
	T_wit_lst = T_k*f_k*mu_k/mu
	T_wit_lst[~msk] = 0.0 ## Groups with no members contribute 0 to inequality
	## Between-group inequalities
	X_k = np.array([np.mean(i) for i in x_lst])
	T_bet = theil_index(X_k[msk],f_k[msk],x_is_group_av=True) ## Using the mask to deal with empty lists
	return np.nansum(T_wit_lst)+T_bet,T_bet,T_wit_lst,T_k

def theil_index(x,w=None,x_is_group_av=True):
	"""
	https://en.wikipedia.org/wiki/Theil_index
	Theil index is a recursive measure by nature, so it might be interesting
	to implement a recursive version for a population divided into smaller
	and smaller groups according to a hierarchical tree of arbitrary depth
	like in Section 3 of https://papers.ssrn.com/abstract=228703
	"""
	x = np.array(x)
	if np.any(x<0):
		raise ValueError("Negative values found in sample. Theil index is not defined for negative values.")
	if w is None:
		mu = np.mean(x)
		N = len(x)
		## We can have 0s in the sample as per https://doi.org/10.1177/2378023115627462
		T = np.nansum(x*np.log(x/mu)) ## nansum counts 0*log(0) as 0
		return T/(N*mu)
	else:
		w = np.array(w)
		assert np.abs(sum(w) - 1.0) < 1e-13
		## If x stores the group averages
		if x_is_group_av:
			mu = np.dot(x,w)
			T = np.nansum(w*x*np.log(x/mu))
			return T/mu
		## If x stores the total (or relative) wealth of each group
		## (notice that in this case x_i would not be scaled by group size)
		## [Explained in The Young Person’s Guide to the Theil Index
		## https://papers.ssrn.com/abstract=228703 where our x is called w and
		## our w is called n
		## and p. 6-7 of fairness metrics notes]
		else:
			x = x/np.sum(x) ## Convert x to proportion of wealth
			return np.dot(x,np.log(x/w))

def CL_delta_groups_1v1(x_lst):
	"""
	Pairwise differences
	"""
	CL_delta_ij = np.zeros((len(x_lst), len(x_lst))) + np.nan
	for i, xi in enumerate(x_lst):
		for j, xj in enumerate(x_lst):
			CL_delta_ij[i,j] = common_language_delta(xi,xj)
	return CL_delta_ij

def CL_delta_groups_1vRest(x_lst):
	"""
	Stochastic homogeneity (p.22 of doi:10.2307/1165329)
	Statistical disparity measure for N groups with each group represented by
	a list of elements containing the wealth of each individual.

	Paramteres
	----------
	x_lst: list
		List of lists, where each list contains the wealth of the individuals 
		of a group.

	Returns
	-------
	result: numpy.ndarray
		Each element i of this array is the difference A-B computed as follows: 
		Take two random individuals, one from group i and another from any of the 
		other groups. A is the probability that the individual from i is wealthier
		than the individual from any of the other groups, and B is the probability
		that the individual from any of the other groups is welathier.
		So a positive value indicates that a given group is on average wealthier
		than any of the other groups and vice-versa.
	"""
	CL_delta_i = np.zeros(len(x_lst)) + np.nan
	for i, xi in enumerate(x_lst):
		x_rest = []
		for j, xj in enumerate(x_lst):
			if i!=j:
				x_rest.extend(xj)
		CL_delta_i[i] = common_language_delta(xi,x_rest)
	return CL_delta_i

def common_language_delta(x,y):
	"""
	doi: 10.2307/1165329
	"""
	A12 = common_language_A12(x,y)
	return 2*A12 - 1

def common_language_A12(x,y):
	"""
	doi: 10.2307/1165329
	See [1] for getting R1 from U1
	[1] https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test#Calculations
	"""
	m = len(x)
	n = len(y)
	if m==0 or n==0:
		return np.nan
	res = stats.mannwhitneyu(x, y)
	U1 = res[0]
	R1 = U1 + 0.5*m*(m+1)
	A12 = (R1/m - (m+1)/2)/n
	return A12

#############################################################################
#############################################################################
## TESTS
#############################################################################
#############################################################################

#############################################################################
## Theil
#############################################################################

def test_theil_index_pareto(realiz=10):
	compare_vals = []
	for _ in range(realiz):
		alf = (np.random.random()*2+1)
		x = stats.pareto.rvs(alf,size=1000000)
		T_emp = theil_index(x)
		T_th = theil_pareto(alf)
		compare_vals.append((T_emp,T_th,100*(T_emp-T_th)/T_th))
	print ("Example:",T_emp,T_th,100*(T_emp-T_th)/T_th)
	return compare_vals

def test_theil_index_uniform(realiz=10):
	T_th = np.log(2/(3*np.sqrt(np.e))) + np.log(2)*2**2/(2**2-1)
	compare_vals = []	
	for _ in range(realiz):
		x = np.random.random(size=1000000)+1
		T_emp = theil_index(x)
		compare_vals.append((T_emp,T_th,100*(T_emp-T_th)/T_th))
	print ("Example:",T_emp,T_th,100*(T_emp-T_th)/T_th)
	return compare_vals

def theil_pareto(alf):
	return np.log(1-1.0/alf) + 1.0/(alf-1)

def test_theil_index_groups(x_lst):
	x_all = []
	for xi in x_lst:
		x_all.extend(xi)
	print ("groups",theil_index_groups(x_lst))
	print ("all",theil_index(x_all))
	# assert theil_index_groups(x_lst)[0] == theil_index(x_all)

#############################################################################
## A12
#############################################################################

def test_A12_p106_VarDel(size1=1000,size2=200):
	x_dom = [1,2,3] 
	x1_p = [.1,.6,.3]
	x2_p = [.3,.6,.1]

	smpl1 = np.random.choice(x_dom,p=x1_p,size=size1)
	smpl2 = np.random.choice(x_dom,p=x2_p,size=size2)

	print("Should be ~.66: ", common_language_A12(smpl1,smpl2))
	print("Should be ~.32: ", common_language_delta(smpl1,smpl2))

def test_A12_p106_VarDel_randomized(size1=1000,size2=200):
	x_dom = [1,2,3] 
	x1_p = np.random.random(size=3)
	x1_p = x1_p / np.sum(x1_p)
	x2_p = np.random.random(size=3)
	x2_p = x2_p / np.sum(x2_p)
	
	smpl1 = np.random.choice(x_dom,p=x1_p,size=size1)
	smpl2 = np.random.choice(x_dom,p=x2_p,size=size2)

	p_gr = x1_p[1]*x2_p[0] + x1_p[2]*x2_p[0] + x1_p[2]*x2_p[1]
	p_less = x2_p[1]*x1_p[0] + x2_p[2]*x1_p[0] + x2_p[2]*x1_p[1]
	p_eq = x2_p[0]*x1_p[0] + x2_p[1]*x1_p[1] + x2_p[2]*x1_p[2]

	print(f"Should be ~{p_gr+0.5*p_eq:.03f}: ", common_language_A12(smpl1,smpl2))
	print(f"Should be ~{p_gr-p_less:.03f}: ", common_language_delta(smpl1,smpl2))
    
    
#############################################################################
## Analytical computation of Common Language (CL) delta measure for ER-style
## networks. DIRECTED VERSION.
#############################################################################

def analytical_multidim_expected_indegree(r,H,F,N):
    """
	Compute expected degree of multidimensional group r in an Erdos-Renyi 
    or SBM type of network given the connection probabilities H, the population
    distribution F, and the number of nodes N.

	Parameters
	----------
	r: tuple
		Attribute vector of group r.
    H: numpy.ndarray
        Matrix of connection probabilities where entry (R,S) is the probability
        of group R to connect to group S.
    F: numpy.ndarray
        Tensor where the element (r1,r2,r3,...,rd) contains the population fraction
        of group r with attribute vector (r1,r2,...,rd).
    N: int
        Number of nodes
	Returns
	-------
	result: float
		Expected in-degree for group r.
	"""
    ## Extract H values as vector
    g_vec = F.shape
    R = comp_index_to_integer(r,g_vec)
    H_col = H[:,R]
    ## Convert F tensor to vector
    comp_indices = make_composite_index(g_vec)
    F_vec = [F[s] for s in comp_indices]
    expected_in_degree = N * np.dot(H_col, F_vec)
    return expected_in_degree

def analytical_multidimensional_delta_num_cdf(r,s,H,F,N,get_probs=False):
    """
	Compute the delta inequality metric for the degree of two 
    groups r, s measuring how much advantage has r over s.
    Notice that the metric is antisymmetric.

	Parameters
	----------
	r: tuple
		Attribute vector of group r.
    s: tuple
        Attribute vector of group s.
    H: numpy.ndarray
        Matrix of connection probabilities where entry (R,S) is the probability
        of group R to connect to group S.
    F: numpy.ndarray
        Tensor where the element (r1,r2,r3,...,rd) contains the population fraction
        of group r with attribute vector (r1,r2,...,rd).
    N: int
        Number of nodes
	Returns
	-------
	result: float
		Value for the delta inequality metric.
	"""
    
    ## Poissonian parameters (expected number of links)
    lambda1 = analytical_multidim_expected_indegree(r,H,F,N)
    lambda2 = analytical_multidim_expected_indegree(s,H,F,N)
    
    ## Cumulative probability for >0 and <0 according to Skellam
    p_upper = skellam.sf(0,lambda1,lambda2) ## p(x1 > x2) = p(x1-x2 > 0) = 1 - p(x1-x2 <= 0)
    p_lower = skellam.cdf(-1,lambda1,lambda2) ## p(x1 < x2) = p(x1-x2 < 0) = p(x1-x2 <= -1)

    if get_probs:
        return p_upper - p_lower, p_upper, p_lower
    else:
        return p_upper - p_lower
    
def analytical_onedim_1vRest_delta_from_multidim_delta(d,x,F,multidim_deltas):
    """
	Compute the one vs rest delta inequality metric for one dimensional groups.

	Parameters
	----------
	d: int
		Dimension index, starting from 0.
    x: int
        Particular one dimensional group or attribute value within dimension d.
    F: numpy.ndarray
        Tensor where the element (r1,r2,r3,...,rd) contains the population fraction
        of group r with attribute vector (r1,r2,...,rd).
    multidim_deltas: dict 
        Dictionary with {key:value} being {(r_vec, s_vec):delta}. Keys are tuples
        of tuples.
	Returns
	-------
	result: float
		Value for the delta inequality metric.
	"""
    g_vec = F.shape
    indices_lst = make_composite_index(g_vec)
    
    norm_constant = 0
    total_delta = 0
    for i_vec in indices_lst:
        for j_vec in indices_lst:
            if i_vec[d] == x and j_vec[d] != x:
                ## VERIFY THIS!! I'm not sure if it is reasonable to 
                ## replace NaN by 0 here!!
                if np.isnan(multidim_deltas[(i_vec, j_vec)]):
                    pass
                else:
                    total_delta += F[i_vec]*F[j_vec]*multidim_deltas[(i_vec, j_vec)]
                    norm_constant += F[i_vec] * F[j_vec]
    
    if total_delta == norm_constant == 0:
        return np.nan
    else:
        return total_delta / norm_constant

def analytical_multidimensional_1vRest_delta_from_multidim_delta(r,F,multidim_deltas):
    """
	Compute the one vs rest delta inequality metric for multidimensional groups.

	Parameters
	----------
	r: tuple
		Attribute vector of group r.
    F: numpy.ndarray
        Tensor where the element (r1,r2,r3,...,rd) contains the population fraction
        of group r with attribute vector (r1,r2,...,rd).
    multidim_deltas: dict 
        Dictionary with {key:value} being {(r_vec, s_vec):delta}. Keys are tuples
        of tuples.
	Returns
	-------
	result: float
		Value for the delta inequality metric.
	"""
    g_vec = F.shape
    indices_lst = make_composite_index(g_vec)
    
    norm_constant = 0
    total_delta = 0
    
    for i_vec in indices_lst:
        if np.all(i_vec != r):
            ## VERIFY THIS!! I'm not sure if it is reasonable to 
            ## replace NaN by 0 here!!
            if np.isnan(multidim_deltas[(r, i_vec)]):
                pass
            else:
                total_delta += F[i_vec] * multidim_deltas[(r, i_vec)]
            norm_constant += F[i_vec]
    
    return total_delta / norm_constant

def analytical_1v1_multidimensional_deltas(H,F,N,get_probs=False):
    """
	Compute all the one vs one delta inequality metric for one-dimensional groups.

	Parameters
	----------
	H: numpy.ndarray
        Matrix of connection probabilities where entry (R,S) is the probability
        of group R to connect to group S.
    F: numpy.ndarray
        Tensor where the element (r1,r2,r3,...,rd) contains the population fraction
        of group r with attribute vector (r1,r2,...,rd).
    N: int 
        Number of nodes.
	Returns
	-------
	result: dict
		{(r,s):delta}
	"""
    g_vec = F.shape
    indices_lst = make_composite_index(g_vec)
    
    multidim_deltas = {}
    if get_probs:
        multidim_deltas_upper = {}
        multidim_deltas_lower = {}
    for r in indices_lst:
        for s in indices_lst:
            if get_probs:
                multidim_deltas[(r, s)], multidim_deltas_upper[(r,s)], multidim_deltas_lower[(r,s)] = analytical_multidimensional_delta_num_cdf(r,s,H,F,N,get_probs=get_probs)
            else:
                multidim_deltas[(r, s)] = analytical_multidimensional_delta_num_cdf(r,s,H,F,N,get_probs=get_probs)
    
    if get_probs:
        return multidim_deltas, multidim_deltas_upper, multidim_deltas_lower
    else:
        return multidim_deltas

def analytical_1vRest_onedimensional_deltas(H,F,N):
    """
	Compute all the one vs rest delta inequality metric for one-dimensional groups.

	Parameters
	----------
	H: numpy.ndarray
        Matrix of connection probabilities where entry (R,S) is the probability
        of group R to connect to group S.
    F: numpy.ndarray
        Tensor where the element (r1,r2,r3,...,rd) contains the population fraction
        of group r with attribute vector (r1,r2,...,rd).
    N: int 
        Number of nodes.
	Returns
	-------
	result: dict
		{r:delta}
	"""
    multidim_deltas = analytical_1v1_multidimensional_deltas(H,F,N)
    
    ndim = F.ndim
    g_vec = F.shape
    
    onedim_deltas_1vRest = {}
    for d in range(ndim):
        onedim_deltas_1vRest[d] = {}
        for vi in range(g_vec[d]):
            onedim_deltas_1vRest[d][vi] = analytical_onedim_1vRest_delta_from_multidim_delta(d,vi,F,multidim_deltas)
    
    return onedim_deltas_1vRest

def analytical_1vRest_multidimensional_deltas(H,F,N):
    """
	Compute all the one vs rest delta inequality metric for multidimensional groups.

	Parameters
	----------
	H: numpy.ndarray
        Matrix of connection probabilities where entry (R,S) is the probability
        of group R to connect to group S.
    F: numpy.ndarray
        Tensor where the element (r1,r2,r3,...,rd) contains the population fraction
        of group r with attribute vector (r1,r2,...,rd).
    N: int 
        Number of nodes.
	Returns
	-------
	result: dict
		{dimension:{attribute_value:delta}}
	"""
    multidim_deltas = analytical_1v1_multidimensional_deltas(H,F,N)
    
    g_vec = F.shape
    indices_lst = make_composite_index(g_vec)
    
    multidim_deltas_1vRest = {}
    for r in indices_lst:
        multidim_deltas_1vRest[r] = analytical_multidimensional_1vRest_delta_from_multidim_delta(r,F,multidim_deltas)

    return multidim_deltas_1vRest
    