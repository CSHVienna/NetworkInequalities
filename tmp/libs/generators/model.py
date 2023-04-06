################################################################
# Dependencies
################################################################

import numpy as np

from tmp.libs.generators import PAH as mPAH, DPAH as mDPAH, DPA as mDPA, DH as mDH, PATCH as mPATCH

################################################################
# Constants
################################################################

MODEL_NAMES = ['PAH', 'PATCH', 'DPAH', 'DPA', 'DH']
MODEL_ACTIVITY_DENSITY = ['DPA','DH','DPAH']
MODEL_BA = ['PAH','PATCH']

MODEL_BASE_H = ['DH']
MODEL_BASE_PA = ['DPA']
MODEL_BASE_PAH = ['PAH','DPAH']
MODEL_BASE_PATCH = ['PATCH']
  
EPSILON = 1e-10

################################################################
# Generators' contructors
################################################################

def PAH(N=100, m=2, fm=0.1, h_MM=0.5, h_mm=0.5, d=None, seed=None):
  G = mPAH.create(N=N, m=m, fm=fm, h_MM=h_MM, h_mm=h_mm, d=d, seed=seed)
  set_metadata(G, N=N, m=m, fm=fm, h_MM=h_MM, h_mm=h_mm, d=d, seed=seed)
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
  name = obj['model'] if 'model' in obj else obj['name']
  if name == 'PAH':
    return PAH(N=obj['N'], m=obj['m'], fm=obj['fm'], h_MM=obj['h_MM'], h_mm=obj['h_mm'], d=obj['d'], seed=seed)
  if name == 'PATCH':
    return PATCH(N=obj['N'], m=obj['m'], fm=obj['fm'], h_MM=obj['h_MM'], h_mm=obj['h_mm'], tc=obj['tc'], seed=seed)
  if name == 'DPAH':
    return DPAH(N=obj['N'], d=obj['d'], fm=obj['fm'], h_MM=obj['h_MM'], h_mm=obj['h_mm'], plo_M=obj['plo_M'], plo_m=obj['plo_m'], seed=seed)
  if name == 'DPA':
    return DPA(N=obj['N'], d=obj['d'], fm=obj['fm'], plo_M=obj['plo_M'], plo_m=obj['plo_m'], seed=seed)
  if name == 'DH':
    return DH(N=obj['N'], d=obj['d'], fm=obj['fm'], h_MM=obj['h_MM'], h_mm=obj['h_mm'], plo_M=obj['plo_M'], plo_m=obj['plo_m'], seed=seed)
  raise Exception("Wrong generator name.")
  
def set_metadata(G, **kwargs):
  for k,v in kwargs.items():
    G.graph[k] = v
    
def get_filename(G, ext=None):
  fn = "_".join([f"{k.replace('_','').replace('name','')}{v}" for k,v in G.graph.items() if k not in ['labels','groups','attributes','class'] and v is not None])
  if ext is not None:
    fn = f"{fn}.{ext}"
  return fn
  
def normalized_homophily_triadic_closure(h_MM, h_mm, tr=None):
  tc = None
  if tr is not None:
    h_MM += EPSILON if h_MM==0 else 0
    h_mm += EPSILON if h_mm==0 else 0
    tr += EPSILON if tr==0 else 0
    h_MM = h_MM / (h_MM+h_mm+tr)
    h_mm = h_mm / (h_MM+h_mm+tr)
    tc = tr / (h_MM+h_mm+tr)
  return h_MM, h_mm, tc
  
def get_empirical_homophily(name, fm, e_MM, e_Mm, e_mM, e_mm, pl_M=None, pl_m=None, pli_M=None, pli_m=None, plo_M=None, plo_m=None, tr=None, verbose=True):
  from tmp.libs.handlers import utils

  if name == 'PAH':
    if verbose:
      utils.warn("Double check.")
    h_MM, h_mm = mPAH.get_empirical_homophily(fm, e_MM, e_Mm, e_mM, e_mm, pl_M, pl_m) # @TODO: verify
  
  elif name == 'PATCH':
    if verbose:
      utils.warn("Double check.")
    h_MM, h_mm = mPAH.get_empirical_homophily(fm, e_MM, e_Mm, e_mM, e_mm, pl_M, pl_m) # @TODO: verify
    h_MM, h_mm, _ = normalized_homophily_triadic_closure(np.float64(h_MM), np.float64(h_mm), tr)
    
  elif name == 'DPAH':
    if verbose:
      utils.warn("Double check.")
    h_MM, h_mm = mDPAH.get_empirical_homophily(fm, e_MM, e_Mm, e_mM, e_mm, pli_M, pli_m, plo_M, plo_m)  # @TODO: verify
  
  elif name == 'DPA':
    if verbose:
      utils.info("DPA model does not require homophily.")
    h_MM, h_mm = None, None
  
  elif name == 'DH':
    h_MM, h_mm = mDH.get_empirical_homophily(fm, e_MM, e_Mm, e_mM, e_mm, plo_M, plo_m)
  
  else:
    raise Exeption("Generator name not valid")
  
  return np.float64(h_MM), np.float64(h_mm)

def get_empirical_triadic_closure(tr, h_MM, h_mm):
  _, _, tc = normalized_homophily_triadic_closure(h_MM, h_mm, tr)
  return tc