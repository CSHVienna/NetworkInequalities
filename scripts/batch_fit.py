############################################################################################################
# Dependencies
############################################################################################################
import os
import argparse

from libs.handlers import utils
from libs.handlers import validations as val
from libs.handlers import io
from libs.handlers import empirical
from libs.generators import model

############################################################################################################
# Functions
############################################################################################################
def fit(args):
  if val.validate_fit_arguments(args):
    
    # empirical
    Ge = io.read_gpickle(args['fn'])
    obj = empirical.get_hyperparams(Ge, args['name'])
    print(obj)
    
#     # fit
#     obj['name'] = args['name']
#     seed = utils.get_random_seed()
#     Gf = model.create(obj, seed=seed)
#     dff = utils.get_node_distributions_as_dataframe(Gf)
    
#     fn = os.path.join(args['output'],args['name'],empirical.get_filename(G))
#     val.validate_path(fn)
#     io.to_gpickle(Gf, f"{fn}.gpickle")
#     io.to_csv(dff, f"{fn}.csv")
    
  else:
    utils.error("Wrong arguments.")

############################################################################################################
# Main
############################################################################################################
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-name",   help="Network generator name", type=str, choices=model.MODEL_NAMES, required=True)
  parser.add_argument("-fn",     help="Path to .gpickle file of empirical network", type=str, required=True)
  parser.add_argument("-output", help="Folder where to store the generated network", type=str, default='../data/fit/', required=True)
  
  args = parser.parse_args()
  args = utils.print_args(args)
  fit(args)
  utils.info("Done!")