############################################################################################################
# Dependencies
############################################################################################################
import pandas as pd
import numpy as np

############################################################################################################
# Constants
############################################################################################################

SMOOTH = 0.1
MINORITY = 1

############################################################################################################
# Computations
############################################################################################################

def apply_rank(df):
  # apply rank per metric (top-k)
  for (name,network_id,metric), tmp in df.groupby(['name','network_id','metric']):
    if not tmp['value'].isnull().values.any():
      df.loc[tmp.index,'rank'] = tmp.loc[:,'value'].rank(method='dense', numeric_only=True, 
                                                         na_option='keep', ascending=False, pct=True)
  return df

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
  if df.shape[0]==0:
    return None
  g = df.groupby(['label'])
  minority = g.filter(lambda x: len(x) == g.size().min()).label.unique()[0]
  return df.query("label==@minority").shape[0] / df.shape[0]

def get_inequity_mean_error(y,fm,smooth=SMOOTH):
  y = np.array(y)
  y = y[y != np.array(None)]
  me = np.mean(y/fm)
  interpretation = 'under-represented' if me<1-smooth else 'over-represented' if me>1+smooth else 'parity'
  return me, interpretation