"""
NetIn is a Python package for the analysis of network inequalities.
It is based on the NetworkX package and provides a set of functions to study inequalities (e.g., in ranking, inference)
in social networks.
"""

__version__ = '1.0.5.8.9'

from netin import generators
from netin.generators import *

from netin import utils
from netin.utils import *

from netin import algorithms
from netin.algorithms import sampling

from netin.generators import convert_networkx_to_netin
