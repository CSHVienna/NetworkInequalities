################################################################
# Dependencies
################################################################
import numpy as np
import pandas as pd
import networkx as nx
from fast_pagerank import pagerank_power

from libs.handlers import validations as val

############################################################################################################
# Prints
############################################################################################################

def _print(kind, txt):
  print(f"[{kind}] {txt}")
  
def info(txt):
  _print('INFO', txt)
  
def warn(txt):
  _print('WARNING', txt)
  
def error(txt):
  _print('ERROR', txt)
  
############################################################################################################
# Ranges
############################################################################################################
def get_homophily_range(h_mm=0.0, h_MM=1.0, steps=0.1):
  val.validate_homophily_range(h_mm, h_MM)
  return [round(v,2) for v in np.arange(h_mm, h_MM+steps, steps)]

def get_fraction_of_minority_range(fm_min=0.1, fm_max=0.5, steps=0.1):
  val.validate_fraction_of_minority_range(fm_min, fm_max)
  return [round(v,2) for v in np.arange(fm_min, fm_max+steps, steps)]

def get_triadic_closure_range(tc_min=0.1, tc_max=1.0, steps=0.2):
  val.validate_triadic_closure_range(tc_min, tc_max)
  return [round(v,2) for v in np.arange(tc_min, tc_max+steps, steps) if v>=tc_min and v<=tc_max]
  
def get_activity_range(beta_min=1.5, beta_max=2.0, steps=0.5):
  val.validate_activity_range(beta_min, beta_max)
  return [round(v,2) for v in np.arange(beta_min, beta_max+steps, steps) if v>=beta_min and v<=beta_max]
  
############################################################################################################
# DataFrames
############################################################################################################
def dataframe_is_empty(df):
  return df.shape[0] == 0

def get_empty_dataframe(columns=[]):
  return pd.DataFrame(columns=columns)

def concat_dataframe(df1, df2, ignore_index=True):
  return pd.concat([df1, df2], ignore_index=ignore_index)

def get_node_distributions_as_dataframe(G, network_id=None):
  N = G.number_of_nodes()
  nodes = G.nodes()
  labels = [G.nodes[n][G.graph['label']] for n in nodes]
  degree = [d for n,d in G.degree(nodes)] if not nx.is_directed(G) else None
  indegree = [d for n,d in G.in_degree(nodes)] if nx.is_directed(G) else None
  outdegree = [d for n,d in G.out_degree(nodes)] if nx.is_directed(G) else None
  A = nx.to_scipy_sparse_matrix(G, nodelist=nodes)
  pagerank = pagerank_power(A)
  
  obj = {k:[v]*N for k,v in G.graph.items() if k not in ['label','groups']}
  
  obj2 = {'network_id':[network_id]*N,
         'node':nodes,
         'label':labels,
         'degree':degree,
         'indegree':indegree,
         'outdegree':outdegree,
         'pagerank':pagerank}
  obj.update(obj2)
  df = pd.DataFrame(obj)
  return df

def flatten_dataframe_by_metric(df):
  data = get_empty_dataframe()
  metrics = ['degree','indegree','outdegree','pagerank']
  for group, tmp in df.groupby(['name','network_id']):
    for metric in metrics:
      tmp2 = tmp.drop(columns=[c for c in metrics if c!=metric]).copy()
      tmp2.rename(columns={metric:'value'}, inplace=True)
      tmp2.loc[:,'metric'] = metric
      tmp2.loc[:,'value'] = tmp2.value.astype(np.float32)
      data = concat_dataframe(data, tmp2)
  return data