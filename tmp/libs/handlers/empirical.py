################################################################
# Dependencies
################################################################
import warnings

warnings.simplefilter("ignore")
warnings.warn("deprecated", RuntimeWarning)

################################################################
# Constants
################################################################

NETWORK_NAMES = ['Caltech','Escorts','GitHub','Swarthmore','Wikipedia','APS','Blogs','Hate','Seventh','Wikipedia2']
PRECISION = 2

################################################################
# Functions
################################################################

def get_filename(G, ext=None):
  fn = "_".join([f"{k.replace('_','').replace('name','')}{v}" for k,v in G.graph.items() if k not in ['label','groups']])
  if ext is not None:
    fn = f"{fn}.{ext}"
  return fn