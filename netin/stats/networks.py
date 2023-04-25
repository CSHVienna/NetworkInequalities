from typing import Union
from typing import Tuple
import warnings

import numpy as np
import networkx as nx
from collections import Counter

from netin.utils import constants as const
from netin.utils import validator as val


def get_min_degree(g: Union[nx.Graph, nx.DiGraph]) -> int:
    degrees = [d for n, d in g.degree]
    return min(degrees) if len(degrees) > 0 else -1


def _get_graph_metadata_value(g: Union[nx.Graph, nx.DiGraph], key: str, default: object = None) -> Union[object, iter]:
    value = default if key not in g.graph or g.graph[key] is None else g.graph[key]
    return value


def _get_class_labels(g: Union[nx.Graph, nx.DiGraph], class_attribute: str = None) -> Tuple[str, str, str]:
    if class_attribute:
        counter = Counter([obj[class_attribute] for n, obj in g.nodes(data=True)]).most_common()
    else:
        val.validate_graph_metadata(g)
        class_attribute = _get_graph_metadata_value(g, 'class_attribute', const.CLASS_ATTRIBUTE)
        counter = Counter([obj[class_attribute] for n, obj in g.nodes(data=True)])

    majority = counter.most_common()[0][0]
    minority = counter.most_common()[-1][0]

    return majority, minority, class_attribute


def get_minority_fraction(g: Union[nx.Graph, nx.DiGraph], class_attribute: str = None) -> float:
    """
    Computes the fraction of the minority class in the graph.

    Parameters
    ----------
    g: Union[nx.Graph, nx.DiGraph]
        Graph to compute the fraction of the minority class

    """
    n = g.number_of_nodes()
    majority, minority, class_attribute = _get_class_labels(g, class_attribute)
    minority_count = sum([1 for n, obj in g.nodes(data=True) if obj[class_attribute] == minority])
    f_m = minority_count / n

    return f_m


def get_edge_type_counts(g: Union[nx.Graph, nx.DiGraph], fractions: bool = False, class_attribute: str = None) -> Counter:
    majority, minority, class_attribute = _get_class_labels(g, class_attribute)
    class_values = [majority, minority]
    class_labels = [const.MAJORITY_LABEL, const.MINORITY_LABEL]

    counts = Counter([f"{class_labels[class_values.index(g.nodes[e[0]][class_attribute])]}"
                      f"{class_labels[class_values.index(g.nodes[e[1]][class_attribute])]}"
                      for e in g.edges])

    if fractions:
        total = sum(counts.values())
        counts = Counter({k: v / total for k, v in counts.items()})

    return counts


def get_average_degree(g: Union[nx.Graph, nx.DiGraph]) -> float:
    k = sum([d for n, d in g.degree]) / g.number_of_nodes()
    return k


def get_average_degrees(g: Union[nx.Graph, nx.DiGraph], class_attribute: str = None) -> Tuple[float, float, float]:
    k = get_average_degree(g)

    majority, minority, class_attribute = _get_class_labels(g, class_attribute)
    kM = np.mean([d for n, d in g.degree if g.nodes[n][class_attribute] == majority])
    km = np.mean([d for n, d in g.degree if g.nodes[n][class_attribute] == minority])

    return k, kM, km


def get_similitude(g: Union[nx.Graph, nx.DiGraph], class_attribute: str = None) -> float:
    majority, minority, class_attribute = _get_class_labels(g, class_attribute)

    tmp = [int(g.nodes[e[0]][class_attribute] == g.nodes[e[1]][class_attribute]) for e in g.edges]
    total = len(tmp)
    sim = sum(tmp) / total

    return sim


def get_node_attributes(g: Union[nx.Graph, nx.DiGraph]) -> list:
    val.validate_graph_metadata(g)
    l = [a for n, a in nx.get_node_attributes(g, g.graph['class_attribute']).items()]
    return l
