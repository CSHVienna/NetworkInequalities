################################################################
# Dependencies
################################################################
import networkx as nx
import powerlaw
from collections import Counter
import numpy as np

from tmp.libs.generators import model
from tmp.libs.handlers import utils

################################################################
# Constants
################################################################

GROUPS = 'groups'
LABELS = 'labels'
CLASS = 'class'
MIN_CLASS = 'm'
MAJ_CLASS = 'M'

################################################################
# Functions
################################################################

def get_mM_class_by_node(G, node):
  return G.graph[GROUPS][G.graph[LABELS].index(G.nodes[node][G.graph[CLASS]])]

def get_minority_class_from_node_counts(G):
  counts = Counter([d[G.graph[CLASS]] for n,d in G.nodes(data=True)])
  print(counts)
  return counts.most_common()[-1][0]
  
def get_sorted_label_classes_Mm(G):
  # c = CLASS if CLASS in G.undigraph else 'label'
  counts = Counter([d[G.graph[CLASS]] for n,d in G.nodes(data=True)])
  print(counts)
  return [counts.most_common()[0][0],counts.most_common()[-1][0]] # M,m

def get_minority_class(G):
  # @TODO: check
  minority_class_index = G.graph[GROUPS].index(MIN_CLASS)
  minority_class = G.graph[LABELS][minority_class_index]
  return minority_class

def get_majority_class(G):
  majority_class_index = G.graph[GROUPS].index(MAJ_CLASS)
  majority_class = G.graph[LABELS][majority_class_index]
  return majority_class

def get_fraction_of_minority(G):
  minority_class = get_minority_class(G)
  n_minorities = sum([obj[G.graph[CLASS]]==minority_class for n,obj in G.nodes(data=True)])
  return n_minorities / G.number_of_nodes()

def get_minimum_degree(G):
  if not G.is_directed():
    return min([d for n,d in G.degree()])
  return None

def get_density(G):
  return nx.density(G)

def get_edge_counts(G):
  counts = Counter([f"{get_mM_class_by_node(G, s)}{get_mM_class_by_node(G, t)}" for s,t in G.edges()])
  e_MM = counts["MM"]
  e_Mm = counts["Mm"]
  e_mM = counts["mM"]
  e_mm = counts["mm"]
  return e_MM, e_Mm, e_mM, e_mm
  
def get_homophily_and_triadic_closure(G, generator_name=model.MODEL_BA[0], verbose=True):
  fm = get_fraction_of_minority(G)
  e_MM, e_Mm, e_mM, e_mm = get_edge_counts(G)
  is_directed = G.is_directed()
  pl, pl_M, pl_m, pli, pli_M, pli_m, plo, plo_M, plo_m = None, None, None, None, None, None, None, None, None
  
  if is_directed:
    plo, plo_M, plo_m = get_powerlaw_exponent_Mm(G, alphas=True, verbose=verbose)
    pli, pli_M, pli_m = get_powerlaw_exponent_Mm(G, indegree=True, alphas=True, verbose=verbose)
  else:
    pl, pl_M, pl_m = get_powerlaw_exponent_Mm(G, alphas=True, verbose=verbose)
  
  try:
    tr = nx.transitivity(G) if generator_name=='PATCH' else None
    h_MM, h_mm = model.get_empirical_homophily(generator_name, fm, e_MM, e_Mm, e_mM, e_mm, pl_M, pl_m, pli_M, pli_m, plo_M, plo_m, tr, verbose)
    tc = model.get_empirical_triadic_closure(tr, h_MM, h_mm)
  except Exception as ex:
    utils.error(f"get_homophily_and_triadic_closure | network.py | {ex}")
    h_MM, h_mm, tc = None, None, None
    
  return h_MM, h_mm, tc

def fit_power_law(x, force=False, discrete=True, verbose=False):
  if force:
    return powerlaw.Fit(x, xmin=min(x), xmax=max(x), discrete=discrete, verbose=verbose)
  return powerlaw.Fit(x, discrete=discrete, verbose=verbose)

def get_powerlaw_exponent_Mm(G, indegree=False, force=True, alphas=False, verbose=True):
  '''
  Returns 3 exponents: 
  - whole distribution
  - distribution only majority nodes
  - distribution only minority nodes
  The distribution is "degree" if the network is undirected, otherwise "outdegree"
  '''
  directed = G.is_directed()
  degrees = G.out_degree() if directed and not indegree else G.in_degree() if directed and indegree else G.degree() if not directed else None
  kind = 'out' if directed and not indegree else 'in' if directed and indegree else ''
  
  if degrees is None:
    raise Exception("Something went wrong.")
    
  if verbose:
    utils.info(f"Power-law fit {kind}degree distribution ({'' if directed else 'un'}directed).")
  
  x = np.array([d for n, d in degrees])
  fit = fit_power_law(x, force=force)

  minority_class = get_minority_class(G)
  majority_class = get_majority_class(G)
  
  x = np.array([d for n, d in degrees if G.nodes[n][G.graph[CLASS]] == minority_class])
  fitm = fit_power_law(x, force=force)

  x = np.array([d for n, d in degrees if G.nodes[n][G.graph[CLASS]] == majority_class])
  fitM = fit_power_law(x, force=force)

  if alphas:
    return fit.alpha, fitM.alpha, fitm.alpha
      
  return fit, fitM, fitm
  
  
def get_hyperparams(G, generator_name, verbose=True):
  
  fm = get_fraction_of_minority(G)
  m = get_minimum_degree(G)
  d = get_density(G)
  pl_M = None
  pl_m = None
  pli_M = None
  pli_m = None
  plo_M = None
  plo_m = None
  h_MM = None
  h_mm = None
  tc = None
  
  h_MM, h_mm, tc = get_homophily_and_triadic_closure(G, generator_name, verbose=verbose)
  
  if G.is_directed():
    plo, plo_M, plo_m = get_powerlaw_exponent_Mm(G, alphas=True, verbose=verbose)
    pli, pli_M, pli_m = get_powerlaw_exponent_Mm(G, indegree=True, alphas=True, verbose=verbose)
  elif not G.is_directed() and (pl_M is None or pl_m is None):
    pl, pl_M, pl_m = get_powerlaw_exponent_Mm(G, alphas=True, verbose=verbose)
    
  obj = {'name':generator_name,
         'N':G.number_of_nodes(),
         'fm':fm, #round(fm,PRECISION), 
         'm':m, 
         'd':d, 
         'h_MM':h_MM, #round(np.float64(h_MM),PRECISION), 
         'h_mm':h_mm, #round(np.float64(h_mm),PRECISION), 
         'tc':tc, 
         'pl_M':pl_M, #round(pl_M,PRECISION), 
         'pl_m':pl_m, #round(pl_m,PRECISION),
         'pli_M':pli_M, #round(pli_M,PRECISION), 
         'pli_m':pli_m, #round(pli_m,PRECISION),
         'plo_M':plo_M, #round(plo_M,PRECISION), 
         'plo_m':plo_m, #round(plo_m,PRECISION)}
        }
  return obj