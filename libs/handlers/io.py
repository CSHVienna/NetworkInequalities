################################################################
# Dependencies
################################################################
import pandas as pd
import os
import glob
import networkx as nx
# import dask.dataframe as dd

from libs.handlers import utils

################################################################
# Functions
################################################################

def exists(fn):
  return os.path.exists(fn)

def read_csv(fn, index_col=0, allow_empty=True):
  if exists(fn):
    df = None
    try:
      # if scalable:
      #   df = dd.read_csv(fn)
      # else:
      df = pd.read_csv(fn, index_col=index_col)
    except Exception as ex:
      utils.error(f"read_csv | io.py | {ex}")
  elif allow_empty:
    df = utils.get_empty_dataframe()
  return df
  
def to_csv(df, fn, index=True, verbose=False):
  try:
    df.to_csv(fn, index=index)
    if verbose:
      utils.info(f"{fn} saved!")
  except Exception as ex:
    utils.error(f"to_csv | io.py | {ex}")
    
def read_gpickle(fn):
  G = None
  try:
    G = nx.read_gpickle(fn)
  except Exception as ex:
    utils.error(f"read_gpickle | io.py | {ex}")
  return G
  
def to_gpickle(G, fn, verbose=False):
  try:
    nx.write_gpickle(G, fn)
    if verbose:
      utils.info(f"{fn} saved!")
  except Exception as ex:
    utils.error(f"to_gpickle | io.py | {ex}")
    