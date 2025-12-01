#Dependencies

from netin import PA
from netin import PAH
from netin import PATC
from netin import PATCH
#from netin import TCH
from netin import viz
from netin import stats

#Network parameters

n = 1000
k = 2
f_m = 0.1
h_MM = 0.9
h_mm = 0.9
tc = 0.8
seed = 1234

#**Generation** 

# PA: Preferential attachment only
g_pa = PA(n=n, k=k, f_m=f_m, seed=seed)
g_pa.generate()
g_pa.info()

###
g_pa.nodes[3]

# PAH: Preferential attachment and homophily
g_pah = PAH(n=n, k=k, f_m=f_m, h_MM=h_MM, h_mm=h_mm, seed=seed)
g_pah.generate()
g_pah.info()

# PATC: Preferential attachment and triadic closure
g_patc = PATC(n=n, k=k, f_m=f_m, tc=tc, seed=seed)
g_patc.generate()
g_patc.info()

# PATCH: Preferential attachment, homophily, and triadic closure
g_patch = PATCH(n=5000, k=k, f_m=f_m, h_MM=h_MM, h_mm=h_mm, tc=tc,seed=seed)
g_patch.generate()
g_patch.info()

# PATCH: Preferential attachment, homophily, and triadic closure
g_patch = PATCH(n=n, k=k, f_m=f_m, h_MM=h_MM, h_mm=h_mm, tc=tc, tc_uniform=False, seed=seed)
g_patch.generate()
g_patch.info()

#def TCH(n, k, f_m, h_MM, h_mm, tc, seed):
#    raise NotImplementedError

# PA: Preferential attachment only
g_tch = TCH(n=n, k=k, f_m=f_m, h_MM=h_MM, h_mm=h_mm, tc=tc, seed=seed)
g_tch.generate()
g_tch.info()

#Plots

g_pa = PA(n=1000, k=2, f_m=0.1, seed=1234)
g_pa.generate()

tc = TCH(g_pa, tc=0.8)  # pass the existing graph and closure probability
tc.apply()  # modifies g_pa in place
g_pa.info()


viz.reset_style()
viz.set_paper_style()

graphs = [g_pa, g_pah, g_patc, g_patch]
viz.plot_graph(graphs, cell_size=2, share_pos=False)
