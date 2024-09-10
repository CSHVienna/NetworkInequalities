from typing import \
    Dict

import os
import glob
import json
import pickle
import networkx as nx
from typing import Union, Tuple


def path_join(*paths):
    return os.path.join(*paths)

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

def save_dict_to_file(dict:Dict, fn):
    with open(fn, 'w') as file:
        json.dump(dict, file, indent=4)

def load_dict_from_file(fn):
    with open(fn, 'r') as file:
        return json.load(file)

def validate_dir(dir:str) -> bool:
    try:
        if not os.path.exists(dir):
            os.makedirs(dir)
    except Exception as ex:
        print(ex)
        return False
    return True