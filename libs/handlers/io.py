################################################################
# Dependencies
################################################################
import pandas as pd
import os
import glob

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