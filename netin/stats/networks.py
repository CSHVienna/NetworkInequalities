from typing import Union
from typing import Tuple
import warnings

import numpy as np
import networkx as nx
from collections import Counter

from netin.utils import constants as const
from netin.utils import validator as val


def get_min_degree(g: Union[nx.Graph, nx.DiGraph]) -> int:
    """
    Returns the minimum degree of nodes in the graph.

    Parameters
    ----------
    g: Union[nx.Graph, nx.DiGraph]
        Graph to compute the minimum degree

    Returns
    -------
    int
        Minimum degree of nodes in the graph
    """
    degrees = [d for n, d in g.degree]
    return min(degrees) if len(degrees) > 0 else -1


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


def get_edge_type_counts(g: Union[nx.Graph, nx.DiGraph], fractions: bool = False,
                         class_attribute: str = None) -> Counter:
    """
    Computes the edge type counts of the graph using the `class_attribute` of each node.

    Parameters
    ----------
    g: Union[nx.Graph, nx.DiGraph]
        Graph to compute the edge type counts

    fractions: bool
        If True, the counts are returned as fractions of the total number of edges.

    class_attribute: str
        The name of the attribute that holds the class label of each node.

    Returns
    -------
    Counter
        Counter holding the edge type counts

    Notes
    -----
    Class labels are assumed to be binary. The minority class is assumed to be labeled as 1.
    """
    majority, minority, class_attribute = _get_class_labels(g, class_attribute)
    class_values = [majority, minority]
    class_labels = [const.MAJORITY_LABEL, const.MINORITY_LABEL]

    counts = Counter([f"{class_labels[class_values.index(g.nodes[e[0]][class_attribute])]}"
                      f"{class_labels[class_values.index(g.nodes[e[1]][class_attribute])]}"
                      for e in g.edges if g.nodes[e[0]][class_attribute] in class_values and
                      g.nodes[e[1]][class_attribute] in class_values])

    if fractions:
        total = sum(counts.values())
        counts = Counter({k: v / total for k, v in counts.items()})

    return counts


def get_average_degree(g: Union[nx.Graph, nx.DiGraph]) -> float:
    """
    Returns the average node degree of the graph.

    Parameters
    ----------
    g: Union[nx.Graph, nx.DiGraph]
        Graph to compute the average degree for

    Returns
    -------
    float
        Average degree of the graph
    """
    k = sum([d for n, d in g.degree]) / g.number_of_nodes()
    return k


def get_average_degrees(g: Union[nx.Graph, nx.DiGraph], class_attribute: str = None) -> Tuple[float, float, float]:
    """
    Computes and returns the average degree of the graph, the average degree of the majority and the minority class.

    Parameters
    ----------
    g: Union[nx.Graph, nx.DiGraph]
        Graph to compute the average degree for

    class_attribute: str
        Name of the class attribute in the graph

    Returns
    -------
    Tuple[float, float, float]
        Average degree of the graph, the average degree of the majority and the minority class
    """
    k = get_average_degree(g)

    majority, minority, class_attribute = _get_class_labels(g, class_attribute)
    kM = np.mean([d for n, d in g.degree if g.nodes[n][class_attribute] == majority])
    km = np.mean([d for n, d in g.degree if g.nodes[n][class_attribute] == minority])

    return k, kM, km


def get_similitude(g: Union[nx.Graph, nx.DiGraph], class_attribute: str = None) -> float:
    """
    Computes and returns the fraction of same-class edges in the graph.

    Parameters
    ----------
    g: Union[nx.Graph, nx.DiGraph]
        Graph to compute the similitude for

    class_attribute: str
        Name of the class attribute in the graph

    Returns
    -------
    float
        Fraction of same-class edges in the graph
    """
    majority, minority, class_attribute = _get_class_labels(g, class_attribute)

    tmp = [int(g.nodes[e[0]][class_attribute] == g.nodes[e[1]][class_attribute]) for e in g.edges]
    total = len(tmp)
    sim = sum(tmp) / total

    return sim


def get_node_attributes(g: Union[nx.Graph, nx.DiGraph]) -> list:
    """
    Returns the values of the class attribute for all nodes in the graph.

    Parameters
    ----------
    g: Union[nx.Graph, nx.DiGraph]
        Graph to get the node attributes from

    Returns
    -------
    list
        List of node attributes
    """
    val.validate_graph_metadata(g)
    l = [a for n, a in nx.get_node_attributes(g, g.graph['class_attribute']).items()]
    return l


def _get_graph_metadata_value(g: Union[nx.Graph, nx.DiGraph], key: str, default: object = None) -> Union[object, iter]:
    value = default if key not in g.graph or g.graph[key] is None else g.graph[key]
    return value


def _get_class_labels(g: Union[nx.Graph, nx.DiGraph], class_attribute: str = None) -> Tuple[str, str, str]:
    if class_attribute:
        counter = Counter([obj[class_attribute] for n, obj in g.nodes(data=True)])
    else:
        val.validate_graph_metadata(g)
        class_attribute = _get_graph_metadata_value(g, 'class_attribute', const.CLASS_ATTRIBUTE)
        counter = Counter([obj[class_attribute] for n, obj in g.nodes(data=True)])

    if len(counter) > 2:
        warnings.warn(f'Graph contains more than two classes: {counter}')

    majority = counter.most_common()[0][0]
    minority = counter.most_common()[1][0]

    return majority, minority, class_attribute
