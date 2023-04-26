import os
import glob
import pickle
import networkx as nx
from typing import Union, Tuple


def read_graph(fn: str) -> Union[nx.Graph, nx.DiGraph]:
    """
    Loads a graph from a file.

    Parameters
    ----------
    fn: str
        Path to file

    Returns
    -------
    Union[nx.Graph, nx.DiGraph]
        Graph
    """
    if fn.endswith('.gml'):
        return nx.read_gml(fn)
    elif fn.endswith('.pkl') or fn.endswith('.gpickle'):
        with open(fn, 'rb') as f:
            return pickle.load(f)
    else:
        raise ValueError(f'Unsupported file format: {fn}')
