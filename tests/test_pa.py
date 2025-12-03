#import unittest 

from netin.models import HomophilyModel, PAModel, PAHModel, PATCHModel, CompoundLFM

# Network parameters
N = 1000
m = 2
f_m = 0.1
seed = 1234

#**Generation** 

# Generation: Preferential attachment
m_pa = PAModel(n=N, k=m, f_m=f_m, seed=seed)
m_pa.simulate()

###
#m_pa.nodes[3]
print(m_pa)