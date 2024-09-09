import pytest
import numpy as np
import networkx as nx

from netin.multidim.generate import multidimensional_network
from netin.multidim.generate.two_dimensional_population import consol_comp_pop_frac_tnsr
from netin.multidim.metrics import network_statistics
from statsmodels.stats.proportion import proportion_confint

class TestMultiNet(object):

	def test_multi_net_case1(self):

		## List of 1d homophily matrices (2 for a two-dimensional system)
		h_mtrx_lst = [ 
		    np.array([[0.9,0.1],
		              [0.1,0.9]]),
		    np.array([[0.6,0.4],
		              [0.4,0.6]])	
		]

		## The marginals of the population distribution defined by comp_pop_frac_tnsr
		## Each row has to sum 1 (100% of the population)
		pop_fracs_lst = [
		    [0.2,0.8],
		    [0.4,0.6]
		]

		## Generate population distribution with certain level of corrrelation
		## No correlation would correspond to the fraction of the largest minority
		consol = 0.4 ## Level of correlation
		comp_pop_frac_tnsr = consol_comp_pop_frac_tnsr(pop_fracs_lst,consol)

		N = 200 ## Number of nodes
		m = 20  ## Average number of connections per node

		kind = "all" ## Aggregation function: {all->and, one->mean, any->or}
		p_d = [0.5, 0.5] ## Weight of each dimension for "mean" aggregation function

		G = multidimensional_network.multidimensional_network_fix_av_degree(
		                h_mtrx_lst,
		                comp_pop_frac_tnsr,
		                kind,
		                directed=False, ## Directed or undirected network
		                pop_fracs_lst = pop_fracs_lst,
		                N=N,
		                m=m,
		                v = 0,
		                p_d = p_d
		                )

		## check network size
		assert G.order() == N
		assert G.number_of_edges() == N*m

		## check number of nodes of each multidimensional type
		multidim_pop, one_dim_pops = network_statistics.comp_group_cnt_tnsr_from_G(G,[2,2])

		for i in range(multidim_pop.shape[0]):
			for j in range(multidim_pop.shape[1]):
				l,u = proportion_confint(
		                        multidim_pop[i,j],
		                        N,
		                        alpha = 0.01,
		                        method="wilson")
				assert l <= comp_pop_frac_tnsr[i,j] <= u

		for d in range(len(one_dim_pops)):
			for i in range(len(one_dim_pops[d])):
				l,u = proportion_confint(
		                        one_dim_pops[d][i],
		                        N,
		                        alpha = 0.01,
		                        method="wilson")
				assert l <= pop_fracs_lst[d][i] <= u