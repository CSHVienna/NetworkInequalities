################################################################
# Dependencies
################################################################
import os

from libs.handlers import utils

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
  flag = False
  for k,v in kws.items():
    if v is None:
      utils.warn(f"{k} is None")
      flag=True
  if flag:
    raise ValueException("[ERROR] validations.py | validate_not_none | At least one parameter is None")
    
def validate_fraction_of_minority_range(fm_min, fm_max):
  if fm_min > 0.0 and fm_min < fm_max and fm_max <= 0.5:
    return True
  raise ValueException("[ERROR] validations.py | validate_fraction_of_minority_range | Wrong values for fm_min and fm_max")
  
def validate_homophily_range(h_mm, h_MM):
  if (h_mm >= 0.0 and h_mm <= 1.0) and (h_MM >= 0.0 and h_MM <= 1.0):
    return True
  raise ValueException("[ERROR] validations.py | validate_homophily_range | Wrong values for h_mm and h_MM")
  
def validate_triadic_closure_range(tc_min, tc_max):
  if tc_min > 0.0 and tc_min < tc_max and tc_max <= 1.0:
    return True
  raise ValueException("[ERROR] validations.py | validate_triadic_closure_range | Wrong values for tc_min and tc_max")
  
def validate_activity_range(beta_min, beta_max):
  if beta_min > 0.0 and beta_min < beta_max:
    return True
  raise ValueException("[ERROR] validations.py | validate_activity_range | Wrong values for beta_min and beta_max")
  
def validate_displot_kind(kind):
  if kind in ['hist','kde','ecdf']:
    return True
  raise ValueException("[ERROR] validations.py | validate_displot_kind | Wrong values for kind")