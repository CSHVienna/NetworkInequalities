from collections import defaultdict
from typing import Union

import networkx as nx
import numpy as np
import powerlaw

from netin.utils import constants as const
from netin.utils import validator as val
from .graph import Graph


class DiGraph(Graph, nx.DiGraph):
    """Directed graph."""
