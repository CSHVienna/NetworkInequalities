################################################################
# Dependencies
################################################################
import os
import numpy as np

from libs.handlers import utils
from libs.handlers.empirical import NETWORK_NAMES
from libs.generators.model import MODEL_NAMES

NAN = ['',' ',None,np.nan]

################################################################
# Functions
################################################################

def validate_path(path):
  try:
    _path = path if os.path.isdir(path) else os.path.dirname(path)
    os.makedirs(_path, exist_ok=True)
    return path
  except Exception as ex:
    utils.error(f"validate_path | validations.py | {ex}")
    return None
  
def validate_not_none(**kws):
  error = False
  for k,v in kws.items():
    if v is None:
      utils.warn(f"{k} is None")
      error=True
  if error:
    raise ValueError("[ERROR] validations.py | validate_not_none | At least one parameter is None")
  return True

def validate_fraction_of_minority_range(fm_min, fm_max):
  if fm_min > 0.0 and fm_min < fm_max and fm_max <= 0.5:
    return True
  raise ValueException("[ERROR] validations.py | validate_fraction_of_minority_range | Wrong values for fm_min and fm_max")
  
def validate_homophily_range(h_mm, h_MM):
  if (h_mm >= 0.0 and h_mm <= 1.0) and (h_MM >= 0.0 and h_MM <= 1.0):
    return True
  raise ValueError("[ERROR] validations.py | validate_homophily_range | Wrong values for h_mm and h_MM")
  
def validate_triadic_closure_range(tc_min, tc_max):
  if tc_min > 0.0 and tc_min < tc_max and tc_max <= 1.0:
    return True
  raise ValueError("[ERROR] validations.py | validate_triadic_closure_range | Wrong values for tc_min and tc_max")
  
def validate_activity_range(beta_min, beta_max):
  if beta_min > 0.0 and beta_min < beta_max:
    return True
  raise ValueError("[ERROR] validations.py | validate_activity_range | Wrong values for beta_min and beta_max")
  
################################################################
# Plots
################################################################

def validate_displot_kind(kind):
  if kind in ['hist','kde','ecdf']:
    return True
  raise ValueError("[ERROR] validations.py | validate_displot_kind | Wrong values for kind")
  
def validate_empirical_vs_fit_kind(kind):
  if kind in ['distributions','inequality','inequity','disparity']:
    return True
  raise ValueError("[ERROR] validations.py | validate_empirical_vs_fit_kind | Wrong values for kind")

################################################################
# Arguments (command line)
################################################################

def validate_create_arguments(obj):
  if 'name' not in obj:
    raise Exception("Generator name is missing")
  if 'output' not in obj:
    raise Exception("Output folder is missing")
  
  if obj['name'] not in MODEL_NAMES:
    raise ValueError("Generator name is not valid")
  
  if obj['name'] == 'PAH':
    if 'N' in obj and 'fm' in obj and 'm' in obj and 'hMM' in obj and 'hmm' in obj:
      return validate_not_none(**{k:v for k,v in obj.items() if k in ['N','fm','m','hMM','hmm']})
  if obj['name'] == 'PATCH':
    if 'N' in obj and 'fm' in obj and 'm' in obj and 'hMM' in obj and 'hmm' in obj and 'tc' in obj:
      return validate_not_none(**{k:v for k,v in obj.items() if k in ['N','fm','m','hMM','hmm','tc']})
  if obj['name'] == 'DPAH':
    if 'N' in obj and 'fm' in obj and 'hMM' in obj and 'hmm' in obj and 'd' in obj and 'ploM' in obj and 'plom' in obj:
      return validate_not_none(**{k:v for k,v in obj.items() if k in ['N','fm','d','hMM','hmm','ploM','plom']})
  if obj['name'] == 'DPA':
    if 'N' in obj and 'fm' in obj and 'd' in obj and 'ploM' in obj and 'plom' in obj:
      return validate_not_none(**{k:v for k,v in obj.items() if k in ['N','fm','d','ploM','plom']})
  if obj['name'] == 'DH':
    if 'N' in obj and 'fm' in obj and 'hMM' in obj and 'hmm' in obj and 'd' in obj and 'ploM' in obj and 'plom' in obj:
      return validate_not_none(**{k:v for k,v in obj.items() if k in ['N','fm','d','hMM','hmm','ploM','plom']})
    
  return False

def validate_fit_arguments(obj):
  if 'name' not in obj:
    raise Exception("Generator name is missing")
  if 'fn' not in obj:
    raise Exception("Root folder is missing")
  if 'output' not in obj:
    raise Exception("Output folder is missing")
  
  if obj['name'] not in MODEL_NAMES:
    raise ValueError("Generator name is not valid")
    
  return True