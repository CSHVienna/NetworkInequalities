############################################################################################################
# Dependencies
############################################################################################################
import os
import glob
import numpy as np
import pandas as pd
import networkx as nx
from fast_pagerank import pagerank_power

from libs.handlers import io
from libs.handlers import validations as val
from libs.generators import model

############################################################################################################
# Constants
############################################################################################################

N_QUANTILES = 10
NODE_METRICS = ['degree','indegree','outdegree','pagerank']
MINORITY = 1
SMOOTH = 0.1

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
  
def print_args(args):
  obj = {}
  print("==================================")
  for arg in vars(args):
    v = getattr(args, arg)
    print(f"{arg}: {v}")
    obj[arg] = v
  print("==================================")
  return obj
  
def get_random_seed():
  return np.random.randint(0,2**32 - 1)

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

def get_rank_range(lb=0.1, ub=1.0, step=0.05):
  return np.arange(lb, ub+step, step)
  
############################################################################################################
# DataFrames
############################################################################################################
def dataframe_is_empty(df):
  return df.shape[0] == 0

def get_empty_dataframe(columns=[]):
  df = pd.DataFrame(columns=columns)
  return df

def concat_dataframe(df1, df2, ignore_index=True):
  try:
    return pd.concat([df1, df2], ignore_index=ignore_index)
  except Exception as ex:
    error(f"utils.py | concat_dataframe | {ex}")
    return None
  
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

def get_main_params(**kws):
  fm = kws['fm'] if 'fm' in kws else None
  m = kws['m'] if 'm' in kws else None
  d = kws['d'] if 'd' in kws else None
  h_MM = kws['h_MM'] if 'h_MM' in kws else None
  h_mm = kws['h_mm'] if 'h_mm' in kws else None
  tc = kws['tc'] if 'tc' in kws else None
  plo_M = kws['plo_M'] if 'plo_M' in kws else None
  plo_m = kws['plo_m'] if 'plo_m' in kws else None
  return fm, m, d, h_MM, h_mm, tc, plo_M, plo_m

def dataframe_sample(data, **kws):
  val.validate_not_none(**kws)
  modelba = model.MODEL_BA
  modelad = model.MODEL_ACTIVITY_DENSITY
  fm,m,d,h_MM,h_mm,tc,plo_M,plo_m = get_main_params(**kws)
  tmp = data.query("fm==@fm and ((m==@m and name in @modelba) or name not in @modelba) and ((h_MM==@h_MM and h_mm==@h_mm and name!='DPA') or name=='DPA') and ((tc==@tc and name=='PATCH') or name!='PATCH') and ((plo_M==@plo_M and plo_m==@plo_m and name in @modelad) or name not in @modelad) and ((d==@d and name in @modelad) or name not in @modelad)").copy()
  return tmp

def flatten_dataframe_by_metric(df):
  data = get_empty_dataframe()
  
  for group, tmp in df.groupby(['name','network_id']):
    for metric in NODE_METRICS:
      tmp2 = tmp.drop(columns=[c for c in NODE_METRICS if c!=metric]).copy()
      tmp2.rename(columns={metric:'value'}, inplace=True)
      tmp2.loc[:,'metric'] = metric
      tmp2.loc[:,'value'] = tmp2.value.astype(np.float32)
      data = concat_dataframe(data, tmp2)
  return data

def apply_rank(df):
  # apply rank per metric (top-k)
  for (name,network_id,metric), tmp in df.groupby(['name','network_id','metric']):
    if not tmp['value'].isnull().values.any():
      df.loc[tmp.index,'rank'] = tmp.loc[:,'value'].rank(method='dense', numeric_only=True, 
                                                                   na_option='keep', ascending=False, pct=True)
  return df

def load_distributions(root, generator_name=None):
  if generator_name is not None:
    df = get_empty_dataframe()
    fn_pattern = os.path.join(root, generator_name, f"{generator_name}_*.csv")
    files = [fn for fn in glob.glob(fn_pattern)]
    for network_id, fn in enumerate(files):
      tmp = io.read_csv(fn)
      tmp.loc[:,'network_id'] = network_id+1
      df = concat_dataframe(df, tmp)
    return df
  else:
    df_results = get_empty_dataframe()
    for name in model.MODEL_NAMES:
      tmp = load_distributions(root, name)
      df_results = concat_dataframe(df_results, tmp)
    return df_results
  return None

############################################################################################################
# Computations
############################################################################################################
def gini(X):
    """Calculate the Gini coefficient of a numpy array."""
    # https://github.com/oliviaguest/gini/blob/master/gini.py
    # based on bottom eq:
    # http://www.statsdirect.com/help/generatedimages/equations/equation154.svg
    # from:
    # http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    # All values are treated equally, arrays must be 1d:
    X = X.flatten()
    if np.amin(X) < 0:
        # Values cannot be negative:
        X -= np.amin(X)
    # Values cannot be 0:
    X += 0.0000001
    # Values must be sorted:
    X = np.sort(X)
    # Index per array element:
    index = np.arange(1, X.shape[0] + 1)
    # Number of array elements:
    n = X.shape[0]
    # Gini coefficient:
    return ((np.sum((2 * index - n - 1) * X)) / (n * np.sum(X)))
  
def percent_minorities(df):
  return df.query("label==@MINORITY").shape[0] / df.shape[0]

def get_inequity_mean_error(y,fm,smooth=SMOOTH):
  y = np.array(y)
  me = np.mean(y/fm)
  interpretation = 'under-represented' if me<1-smooth else 'over-represented' if me>1+smooth else 'parity'
  return me, interpretation