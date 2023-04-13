"""
NetIn is a Python package for the analysis of network inequalities.
It is based on the NetworkX package and provides a set of functions to study inequalities (e.g., in ranking, inference)
in social networks.
"""

__version__ = '1.0.0'

from netin import algorithms
from netin.algorithms import *

from netin import generators
from netin.generators import *
from netin.generators.graph import *
from netin.generators.undigraph import *
from netin.generators.pa import *
from netin.generators.pah import *
from netin.generators.patc import *
from netin.generators.patch import *
from netin.generators.digraph import *
from netin.generators.dpa import *
from netin.generators.dh import *
from netin.generators.dpah import *

# from netin import stats
# from netin.stats import *

# from netin import utils
# from netin.utils import *

# from netin import viz
# from netin.viz import *
