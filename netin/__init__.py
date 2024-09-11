"""
NetIn is a Python package for the analysis of network inequalities.
It provides models to simulate social networks and functions
to study inequalities (e.g., in ranking, inference) in them.
"""

__version__ = '2.0.0a1'

from netin import utils
from netin.utils import *

from netin import algorithms
from netin.algorithms import sampling
from netin.algorithms import janus

from netin import graphs
from netin.graphs import *

from netin import link_formation_mechanisms
from netin import models
from netin import filters
