############################################################################################################
# Dependencies
############################################################################################################
import os
import argparse

from libs.handlers import utils
from libs.handlers import validations as val
from libs.handlers import io
from libs.generators import model

############################################################################################################
# Functions
############################################################################################################
def create(args):
  if val.validate_create_arguments(args):
    
    seed = utils.get_random_seed()
    G = model.create(args, seed=seed)
    df = utils.get_node_distributions_as_dataframe(G)
    
    fn = os.path.join(args['output'],args['name'],model.get_filename(G))
    val.validate_path(fn)
    io.to_gpickle(G, f"{fn}.gpickle")
    io.to_csv(df, f"{fn}.csv")
    
  else:
    utils.error("Wrong arguments.")

############################################################################################################
# Main
############################################################################################################
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-name",   help="Network generator name", type=str, choices=model.MODEL_NAMES, required=True)
  parser.add_argument("-N",      help="Number of nodes", type=int, required=True)
  parser.add_argument("-m",      help="Minimun degree of node (Barabasi-Albert)", type=int, required=False)
  parser.add_argument("-fm",     help="Fraction of minorities", type=float, required=True)
  parser.add_argument("-d",      help="Edge density", type=float, default=None, required=False)
  parser.add_argument("-hMM",    help="Homophily within majority group", type=float, default=None, required=False)
  parser.add_argument("-hmm",    help="Homophily within minority group", type=float, default=None, required=False)
  parser.add_argument("-tc",     help="Triadic closure probability", type=float, default=None, required=False)
  parser.add_argument("-ploM",   help="Power-law exponent for activity distribution of majority nodes", type=float, default=None, required=False)
  parser.add_argument("-plom",   help="Power-law exponent for activity distribution of minority nodes", type=float, default=None, required=False)
  parser.add_argument("-output", help="Folder where to store the generated network.", type=str, default='../data/synthetic/', required=True)
  
  args = parser.parse_args()
  args = utils.print_args(args)
  create(args)
  utils.info("Done!")