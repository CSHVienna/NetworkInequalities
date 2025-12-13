#import unittest 
#Dependencies
from netin.models import HomophilyModel, PAModel, PAHModel, PATCHModel, CompoundLFM
from netin import viz
from netin.stats import get_node_metadata_as_dataframe

# Network parameters
n = 1000
k = 2
f_m = 0.1
h_MM = 0.9
h_mm = 0.9
tc = 0.8
seed = 1234
N = n
#Generation

# PA: Preferential attachment
m_pa = PAModel(n=n, k=k, f_m=f_m, seed=seed)
m_pa.simulate()

# H: Homophily
m_h = HomophilyModel(
    n=n, k=k, f_m=f_m, h_mm=h_mm, h_MM=h_MM, seed=seed
)

# PAH: Preferential attachment and homophily
m_pah = PAHModel(
    n=n, k=k, f_m=f_m, h_mm=h_mm, h_MM=h_MM, seed=seed
)

# PATCH: Preferential attachment, homophily, and triadic closure
lfm_l = CompoundLFM.PAH
lfm_g = CompoundLFM.PAH
tau = 0.0

m_patch = PATCHModel(
    n=n,
    k=k,
    f_m=f_m,
    tau=tau,
    lfm_tc=lfm_l,
    lfm_global=lfm_g,
    h_mm=h_mm,
    h_MM=h_MM,
    seed=seed,
)

m_patch.simulate()

# Plots
viz.reset_style()
viz.set_paper_style()

