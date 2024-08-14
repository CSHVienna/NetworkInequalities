"""
NetIn is a Python package for the analysis of network inequalities.
It is based on the NetworkX package and provides a set of functions to study inequalities (e.g., in ranking, inference)
in social networks.
"""

__version__ = '1.0.8'

from netin import utils
from netin.utils import *

from netin import algorithms
from netin.algorithms import sampling

from netin import graphs
from netin.graphs import *

from netin import link_formation_mechanisms
from netin import models
from netin import filters
