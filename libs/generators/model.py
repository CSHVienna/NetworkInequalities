################################################################
# Dependencies
################################################################

import networkx as nx

from libs.generators import PAH as mPAH
from libs.generators import DH as mDH
from libs.generators import DPA as mDPA
from libs.generators import DPAH as mDPAH
from libs.generators import PATCH as mPATCH
from libs.generators import MAH

MODEL_NAMES = ['PAH', 'PATCH', 'DPAH', 'DPA', 'DH']
MODEL_ACTIVITY_DENSITY = ['DPA','DH','DPAH']
MODEL_BA = ['PAH','PATCH']
  
################################################################
# Generators' contructors
################################################################

def PAH(N=100, m=2, fm=0.1, h_MM=0.5, h_mm=0.5, seed=None):
  G = mPAH.create(N=N, m=m, fm=fm, h_MM=h_MM, h_mm=h_mm, seed=seed)
  set_metadata(G, N=N, m=m, fm=fm, h_MM=h_MM, h_mm=h_mm, seed=seed)
  return G

def PATCH(N=100, m=2, fm=0.1, h_MM=0.5, h_mm=0.5, tc=0.1, seed=None):
  G = mPATCH.create(N=N, m=m, fm=fm, h_MM=h_MM, h_mm=h_mm, triadic_closure=tc, seed=seed)
  set_metadata(G, N=N, m=m, fm=fm, h_MM=h_MM, h_mm=h_mm, tc=tc, seed=seed)
  return G

def DPAH(N=100, fm=0.1, d=0.1, h_MM=0.5, h_mm=0.5, plo_M=0.2, plo_m=0.2, seed=None):
  G = mDPAH.create(N=N, fm=fm, d=d, h_MM=h_MM, h_mm=h_mm, plo_M=plo_M, plo_m=plo_m, seed=seed)
  set_metadata(G, N=N, fm=fm, d=d, h_MM=h_MM, h_mm=h_mm, plo_M=plo_M, plo_m=plo_m, seed=seed)
  return G

def DPA(N=100, fm=0.1, d=0.1, plo_M=0.2, plo_m=0.2, seed=None):
  G = mDPA.create(N=N, fm=fm, d=d, plo_M=plo_M, plo_m=plo_m, seed=seed)
  set_metadata(G, N=N, fm=fm, d=d, plo_M=plo_M, plo_m=plo_m, seed=seed)
  return G

def DH(N=100, fm=0.1, d=0.1, h_MM=0.5, h_mm=0.5, plo_M=0.2, plo_m=0.2, seed=None):
  G = mDH.create(N=N, fm=fm, d=d, h_MM=h_MM, h_mm=h_mm, plo_M=plo_M, plo_m=plo_m, seed=seed)
  set_metadata(G, N=N, fm=fm, d=d, h_MM=h_MM, h_mm=h_mm, plo_M=plo_M, plo_m=plo_m, seed=seed)
  return G

def MABAH():
  return

################################################################
# Functions
################################################################

def create(obj, seed=None):
  if obj['name'] == 'PAH':
    return PAH(N=obj['N'], m=obj['m'], fm=obj['fm'], h_MM=obj['hMM'], h_mm=obj['hmm'], seed=seed)
  if obj['name'] == 'PATCH':
    return PATCH(N=obj['N'], m=obj['m'], fm=obj['fm'], h_MM=obj['hMM'], h_mm=obj['hmm'], tc=obj['tc'], seed=seed)
  if obj['name'] == 'DPAH':
    return DPAH(N=obj['N'], d=obj['d'], fm=obj['fm'], h_MM=obj['hMM'], h_mm=obj['hmm'], plo_M=obj['ploM'], plo_m=obj['plom'], seed=seed)
  if obj['name'] == 'DPA':
    return DPA(N=obj['N'], d=obj['d'], fm=obj['fm'], plo_M=obj['ploM'], plo_m=obj['plom'], seed=seed)
  if obj['name'] == 'DH':
    return DH(N=obj['N'], d=obj['d'], fm=obj['fm'], h_MM=obj['hMM'], h_mm=obj['hmm'], plo_M=obj['ploM'], plo_m=obj['plom'], seed=seed)
  raise Exception("Wrong generator name.")
  
def set_metadata(G, **kwargs):
  for k,v in kwargs.items():
    G.graph[k] = v
    
def get_filename(G, ext=None):
  fn = "_".join([f"{k.replace('_','').replace('name','')}{v}" for k,v in G.graph.items() if k not in ['label','groups']])
  if ext is not None:
    fn = f"{fn}.{ext}"
  return fn
  