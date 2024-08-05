from typing import Union
from typing import Tuple
import warnings

import numpy as np
import networkx as nx
import pandas as pd
from collections import Counter

from netin.utils import constants as const
from netin.utils import validator as val
from ..graphs.graph import Graph
from ..graphs.directed import DiGraph

def get_node_metadata_as_dataframe(graph: Union[Graph, DiGraph], include_graph_metadata: bool = False, n_jobs: int = 1) -> pd.DataFrame:
        """
        Returns the metadata of the nodes in the graph as a dataframe.
        Every row represents a node, and the columns are the metadata of the node.

        Parameters
        ----------
        include_graph_metadata: bool
            whether to include the graph metadata (e.g., class attribute, class values, etc.)

        n_jobs: int
            number of parallel jobs

        Returns
        -------
        pd.DataFrame
            dataframe with the metadata of the nodes

        Notes
        -----
        Column `class_label` is a binary column indicating whether the node belongs to the minority class.
        """
        cols = ['node', 'class_label', 'real_label', 'source']

        obj = {'node': graph.node_list,
               'class_label': [const.MAJORITY_LABEL if graph.get_class_label_by_node(
                   n) == graph.get_majority_label() else const.MINORITY_LABEL for n in graph.node_list],
               'real_label': [graph.get_class_label_by_node(n) for n in graph.node_list],
               'source': 'model' if 'empirical' not in graph.graph else 'data'}

        # include graph metadata
        if include_graph_metadata:
            # n = graph.number_of_nodes()
            new_cols = [c for c in graph.graph.keys() if c not in ['class_attribute', 'class_values',
                                                                  'class_labels']]
            obj.update({c: graph.graph[c] for c in new_cols})
            cols.extend(new_cols)

        # include metrics
        column_values = pqdm(const.VALID_METRICS, graph.compute_node_stats, n_jobs=n_jobs)
        obj.update({col: values for col, values in zip(const.VALID_METRICS, column_values)})
        cols.extend(const.VALID_METRICS)

        # create dataframe
        df = pd.DataFrame(obj, columns=cols, index=graph.node_list)
        df.name = graph.model_name

        # add ranking values
        for metric in const.VALID_METRICS:
            ncol = f'{metric}_rank'
            df.loc[:, ncol] = df.loc[:, metric].rank(ascending=False, pct=True, method='dense')

            # # compute ranking values and retry for ARPACK error.
            # done = False
            # tries = 10
            # while not done:
            #     try:
            #         df.loc[:, ncol] = df.loc[:, metric].rank(ascending=False, pct=True, method='dense')
            #         done = True
            #     except Exception as ex:
            #         tries -= 1
            #         if tries <= 0:
            #             raise UserWarning(f"An error occurred while computing the ranking values: {ex}")

        return df

def compute_node_stats(graph: Union[Graph, DiGraph], metric: str, **kwargs) -> list[Union[int, float]]:
    """
    Returns the property of each node in a graph based on the metric.

    Parameters
    ----------
    metric: str
        metric to compute

    kwargs: dict
        additional parameters for the metric

    Returns
    -------
    List[Union[int, float]]
        list properties for each node
    """
    values = None

    # list of tuples (node, value)
    if metric == 'degree':
        values = graph.degree(graph.node_list, **kwargs) if not graph.is_directed() else None
    if metric == 'in_degree':
        values = graph.in_degree(graph.node_list, **kwargs) if graph.is_directed() else None
    if metric == 'out_degree':
        values = graph.out_degree(graph.node_list, **kwargs) if graph.is_directed() else None
    if metric == 'eigenvector':
        try:
            values = nx.eigenvector_centrality_numpy(graph, **kwargs)
        except Exception as ex:
            try:
                values = nx.eigenvector_centrality_numpy(graph, max_iter=200, tol=1.0e-5)
            except Exception as ex:
                warnings.warn(f"The eigenvector centrality could not be computed: {ex}")
                values = None

    # dict of node -> value
    if metric == 'clustering':
        values = nx.clustering(graph, graph.node_list, **kwargs)
    if metric == 'betweenness':
        values = nx.betweenness_centrality(graph, **kwargs)
    if metric == 'closeness':
        if isinstance(graph, nx.DiGraph):
            values = nx.closeness_centrality(nx.DiGraph(graph), **kwargs)
        else:
            values = nx.closeness_centrality(graph, **kwargs)
    if metric == 'pagerank':
        values = nx.pagerank(graph, **kwargs)

    return [values[n] for n in graph.node_list] if values is not None else np.nan

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
