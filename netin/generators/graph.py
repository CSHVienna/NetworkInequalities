import time
import warnings
from collections import Counter
from typing import Union, Iterable, Any, Dict

import networkx as nx
import numpy as np
import pandas as pd
import powerlaw
from pqdm.threads import pqdm

import netin
from netin.stats import networks as net
from netin.utils import constants as const
from netin.utils import validator as val


class Graph(nx.Graph):
    """Bi-populated network (base graph model)

    Parameters
    ----------
    n: int
        number of nodes (minimum=2)

    f_m: float
        fraction of minorities (minimum=1/n, maximum=(n-1)/n)

    seed: object
        seed for random number generator

    Notes
    -----
    The initialization is a graph with n nodes and fraction of minority f_m.
    Source nodes are selected one-by-one (if undirected) and based on their activity (if directed).
    Target nodes are selected depending on the chosen edge mechanism.

    Available target mechanisms for undirected networks:
    - preferential attachment (pa), see :class:`netin.PA` [1]_.
    - preferential attachment with homophily (h_**), see :class:`netin.PAH` [2]_.
    - preferential attachment with triadic closure (tc), see :class:`netin.PATC` [3]_.
    - preferential attachment with homophily and triadic closure (tc), see :class:`netin.PATCH`.

    Available target mechanisms for directed networks:
    - preferential attachment, see :class:`netin.DPA` [4]_
    - homophily, see :class:`netin.DH` [4]_
    - preferential attachment with homophily (h_**), see :class:`netin.DPAH` [4]_

    References
    ----------
    .. [BarabasiAlbert1999] A. L. Barabasi and R. Albert "Emergence of scaling in random networks", Science 286, pp 509-512, 1999.
    .. [Karimi2018] F. Karimi, M. Génois, C. Wagner, P. Singer, & M. Strohmaier, M "Homophily influences ranking of minorities in social networks", Scientific reports 8(1), 11077, 2018.
    .. [HolmeKim2002] P. Holme and B. J. Kim “Growing scale-free networks with tunable clustering” Phys. Rev. E 2002.
    .. [Espin-Noboa2022] L. Espín-Noboa, C. Wagner, M. Strohmaier, & F. Karimi "Inequality and inequity in network-based ranking and recommendation algorithms" Scientific reports 12(1), 1-14, 2022.
    """

    ############################################################
    # Constructor
    ############################################################

    def __init__(self, n: int, f_m: float, seed: object = None):
        nx.Graph.__init__(self)
        self.n = n  # number of nodes
        self.f_m = f_m  # fraction of minority
        self.seed = seed  # random seed
        self.n_m = 0  # number of minority nodes
        self.n_M = 0  # number of majority nodes
        self.model_name = None  # name of the network model
        self.class_attribute = None  # name of the class attribute (e.g., 'class')
        self.class_values = None  # list of class values (e.g., [0, 1])
        self.class_labels = None  # list of class labels (e.g., ['minority', 'majority'])
        self.node_list = None  # vector of nodes
        self.node_class_values = None  # dictionary of node class values
        self.gen_start_time = None  # start time of the generation
        self.gen_duration = None  # duration of the generation

    ############################################################
    # Init
    ############################################################

    def validate_parameters(self):
        """
        Validates the parameters of the graph (n, f_m, seed).
        """
        val.validate_int(self.n, minimum=2)
        val.validate_float(self.f_m, minimum=1 / self.n, maximum=(self.n - 1) / self.n)
        self.seed = self.seed if self.seed is not None else np.random.randint(0, 2 ** 10)

    def set_class_info(self, class_attribute: str = const.CLASS_ATTRIBUTE,
                       class_values: list = None,
                       class_labels: list = None):
        """
        Sets the class_attribute, class_values and class_labels.

        Parameters
        ----------
        class_attribute: str
            name of the class attribute

        class_values:
            list of class values

        class_labels:
            list of class labels
        """
        if class_labels is None:
            class_labels = const.CLASS_LABELS
        if class_values is None:
            class_values = const.CLASS_VALUES
        self.class_attribute = class_attribute
        self.class_values = class_values
        self.class_labels = class_labels

    def get_metadata_as_dict(self) -> dict:
        """
        Returns metadata for a graph as a dictionary.
        It includes the model name, class attribute, class values, class labels, number of nodes,
        fraction of minority, and seed.

        Returns
        -------
        dict
            metadata for a graph
        """
        obj = {'model': self.model_name,
               'class_attribute': self.class_attribute,
               'class_values': self.class_values,
               'class_labels': self.class_labels,
               'n': self.n,
               'f_m': self.f_m,
               'seed': self.seed}
        return obj

    ############################################################
    # Getters and setters
    ############################################################

    @property
    def n(self) -> int:
        """
        Returns the expected number of nodes (n, the input parameter).

        Returns
        -------
        int
            number of nodes
        """
        return self._n

    @n.setter
    def n(self, n: int):
        """
        Sets the expected number of nodes.

        Parameters
        ----------
        n: int
            number of nodes
        """
        self._n = n

    @property
    def f_m(self) -> float:
        """
        Returns the expected fraction of minorities (f_m, the input parameter).

        Returns
        -------
        float
            fraction of minorities
        """
        return self._f_m

    @f_m.setter
    def f_m(self, f_m: float):
        """
        Sets the expected fraction of minorities.

        Parameters
        ----------
        f_m: float
            fraction of minorities (minimum=1/n, maximum=(n-1)/n)
        """
        self._f_m = f_m

    @property
    def node_list(self) -> np.array:
        """
        Returns the vector of nodes.

        Returns
        -------
        np.array
            vector of nodes
        """
        return self._node_list

    @node_list.setter
    def node_list(self, node_list: Union[np.array, list]):
        """
        Sets the vector of nodes.

        Parameters
        ----------
        nodes: np.array
            vector of nodes
        """
        self._node_list = node_list

    @property
    def node_class_values(self) -> dict[int, Any]:
        """
        Returns the node class values.

        Returns
        -------
        dict
            dictionary of node class values
        """
        return self._node_class_values

    @node_class_values.setter
    def node_class_values(self, node_class_values: dict[int, Any]):
        """
        Sets the node class values.

        Parameters
        ----------
        node_class_values: dict
            dictionary of node class values
        """
        self._node_class_values = node_class_values

    @property
    def n_m(self) -> int:
        """
        Returns the number of minority nodes.

        Returns
        -------
        int
            number of minority nodes
        """
        return self._n_m

    @n_m.setter
    def n_m(self, n_m: int):
        """
        Sets the number of minority nodes.

        Parameters
        ----------
        n_m: int
            number of minority nodes
        """
        self._n_m = n_m

    @property
    def n_M(self) -> int:
        """
        Returns the number of majority nodes.

        Returns
        -------
        int
            number of majority nodes
        """
        return self._n_M

    @n_M.setter
    def n_M(self, n_M: int):
        """
        Sets the number of majority nodes.

        Parameters
        ----------
        n_M: int
            number of majority nodes
        """
        self._n_M = n_M

    @property
    def seed(self) -> object:
        """
        Returns the random seed (seed, the input parameter).

        Returns
        -------
        object
            random seed
        """
        return self._seed

    @seed.setter
    def seed(self, seed=None):
        """
        Sets the random seed.

        Parameters
        ----------
        seed
            random seed
        """
        self._seed = seed

    @property
    def model_name(self) -> str:
        """
        Returns the name of the model used to generate the graph.

        Returns
        -------
        str
            name of the model
        """
        return self._model_name

    @model_name.setter
    def model_name(self, model_name: str):
        """
        Sets the name of the model used to generate the graph.

        Parameters
        ----------
        model_name: str
            name of the model
        """
        self._model_name = model_name

    @property
    def class_attribute(self) -> str:
        """
        Returns the class attribute of the graph used to infer minority and majority nodes.

        Returns
        -------
        str
            name of the class attribute
        """
        return self._class_attribute

    @class_attribute.setter
    def class_attribute(self, class_attribute: str):
        """
        Sets the class attribute of the graph used to infer minority and majority nodes.

        Parameters
        ----------
        class_attribute: str
            name of the class attribute
        """
        self._class_attribute = class_attribute

    @property
    def class_values(self) -> list:
        """
        Returns the class values under the class attribute.

        Returns
        -------
        list
            list of class values
        """
        return self._class_values

    @class_values.setter
    def class_values(self, class_values: list):
        """
        Sets the class values under the class attribute.
        These values are used to identify minority and majority nodes.

        Parameters
        ----------
        class_values: list
            list of class values
        """
        self._class_values = class_values

    @property
    def class_labels(self) -> list:
        """
        Returns the class labels for minority and majority that map the class values.

        Returns
        -------
        list
            list of class labels
        """
        return self._class_labels

    @class_labels.setter
    def class_labels(self, class_labels: list):
        """
        Sets the class labels for minority and majority that map the class values.

        Parameters
        ----------
        class_labels: list
            list of class labels
        """
        self._class_labels = class_labels

    ############################################################
    # Node or graph attributes
    ############################################################

    def get_class_value_by_node(self, node: int) -> int:
        """
        Returns the class value of a given node (id).

        Parameters
        ----------
        node: int
            node id

        Returns
        -------
        int
            class value

        """
        return self.node_class_values[node]

    def get_class_label_by_node(self, node: int) -> str:
        """
        Returns the class label of a given node (id).

        Parameters
        ----------
        node: int
            node id

        Returns
        -------
        str
            class label
        """
        class_value = self.get_class_value_by_node(node)
        return self.class_labels[class_value]

    def get_majority_value(self) -> object:
        """
        Returns the value for the majority class.

        Returns
        -------
        int
            value for the majority class

        Notes
        -----
        Position 0 of the ``class_values`` is the majority value.
        Position 1 of the ``class_values`` is the minority value.
        """
        # return const.MAJORITY_VALUE
        return self.class_values[const.MAJORITY_VALUE]

    def get_minority_value(self) -> object:
        """
        Returns the value for the minority class

        Returns
        -------
        int
            value for the minority class

        Notes
        -----
        Position 0 of the ``class_values`` is the majority value.
        Position 1 of the ``class_values`` is the minority value.
        """
        # return const.MINORITY_VALUE
        return self.class_values[const.MINORITY_VALUE]

    def get_majority_label(self) -> str:
        """
        Returns the label for the majority class.

        Returns
        -------
        str
            label for the majority class

        Notes
        -----
        Position 0 of the ``class_labels`` is the majority label.
        Position 1 of the ``class_labels`` is the minority label.
        """
        # return const.MAJORITY_LABEL
        return self.class_labels[const.MAJORITY_VALUE]

    def get_minority_label(self) -> str:
        """
        Returns the labels for the minority class.

        Returns
        -------
        str
            label for the minority class

        Notes
        -----
        Position 0 of the ``class_labels`` is the majority label.
        Position 1 of the ``class_labels`` is the minority label.
        """
        # return const.MINORITY_LABEL
        return self.class_labels[const.MINORITY_VALUE]

    def get_expected_number_of_edges(self):
        pass

    ############################################################
    # Generation
    ############################################################

    def initialize(self, class_attribute: str = const.CLASS_ATTRIBUTE, class_values: list = None,
                   class_labels: list = None):
        """
        Initializes the random seed, the graph metadata, and node class information.
        """
        np.random.seed(self.seed)
        self.validate_parameters()
        self.init_graph(class_attribute, class_values, class_labels)
        self.init_nodes()

    def init_graph(self, class_attribute: str = const.CLASS_ATTRIBUTE, class_values: list = None,
                   class_labels: list = None):
        """
        Initializes the graph.
        Sets the name of the model, class information, and the graph metadata.

        Parameters
        ----------
        class_attribute: str
            name of the class attribute

        class_values:
            list of class values

        class_labels:
            list of class labels
        """
        self.set_class_info(class_attribute, class_values, class_labels)
        self.graph.update(self.get_metadata_as_dict())

    def init_nodes(self):
        """
        Initializes the list of nodes with their respective labels.
        """
        self.node_list = np.arange(self.n)
        self.n_M = int(round(self.n * (1 - self.f_m)))
        self.n_m = self.n - self.n_M
        minorities = np.random.choice(self.node_list, self.n_m, replace=False)
        self.node_class_values = {n: int(n in minorities) for n in self.node_list}
        self.add_nodes_from(self.node_list)
        nx.set_node_attributes(self, self.node_class_values, self.class_attribute)

    def get_potential_nodes_to_connect(self, source: int,
                                       available_nodes: list[int]) -> list[int]:
        """
        From the list of available_nodes, it filters out the source node and the nodes that are already connected
        to the source node.

        Parameters
        ----------
        source: int
            source node id

        available_nodes: Set[int]
            set of target node ids

        Returns
        -------
        Set[int]
            set of potential target node ids
        """
        return [t for t in available_nodes if t != source and t not in nx.neighbors(self, source)]

    def get_target_probabilities(self, source: int,
                                 available_nodes: list[int] = None) -> tuple[np.array, list[int]]:
        """
        Returns the probability for each target node to be connected to the source node.

        Parameters
        ----------
        source:  int
            source node id (not used here)

        available_nodes:
            list of target node ids

        Notes
        -----
        It follows an Erdos-Renyi model [ErdosRenyi1959]_.

        Returns
        -------
        tuple[np.array, set[int]]
            tuple of probabilities and target set

        References
        ----------
        .. [ErdosRenyi1959] P. Erdős and A. Rényi, On Random Graphs, Publ. Math. 6, 290 (1959).
        """
        # Random (erdos renyi)
        probs = np.ones(len(available_nodes))
        probs /= probs.sum()
        return probs, available_nodes

    def get_target(self, source: int,
                   available_nodes: list[int]) -> int:
        pass

    def on_edge_added(self, source: int, target: int):
        pass

    def generate(self):
        """
        Basic instructions run before generating the graph.
        """
        self.gen_start_time = time.time()

        # init graph and nodes
        self.initialize()

        # iterate
        # for each source_node, get target (based on specific mechanisms), then add edge, update special available_nodes

    def terminate(self):
        """
        Instructions run after the generation of the graph.
        """
        self.gen_duration = time.time() - self.gen_start_time

    ############################################################
    # Calculations
    ############################################################

    def info_params(self):
        pass

    def info_computed(self):
        pass

    def info(self):
        """
        Shows the parameters and the computed properties of the graph.
        """
        print("=== Params ===")
        print('n: {}'.format(self.n))
        print('f_m: {}'.format(self.f_m))
        print(f'number of minority nodes: {self.n_m}')
        self.info_params()
        print('seed: {}'.format(self.seed))

        print("=== Model ===")
        print('Model: {}'.format(self.model_name))
        print('Class attribute: {}'.format(self.class_attribute))
        print('Class values: {}'.format(self.class_values))
        print('Class labels: {}'.format(self.class_labels))
        print('Generation time: {} (secs)'.format(self.gen_duration))

        print("=== Computed ===")
        print(f'- is directed: {self.is_directed()}')
        print(f'- number of nodes: {self.number_of_nodes()}')
        print(f'- number of edges: {self.number_of_edges()}')
        print(f'- expected number of edges: {self.get_expected_number_of_edges()}')

        print(f'- minimum degree: {self.calculate_minimum_degree()}')
        print(f'- fraction of minority: {self.calculate_fraction_of_minority()}')
        print(f'- edge-type counts: {self.calculate_edge_type_counts()}')
        print(f"- density: {nx.density(self)}")
        try:
            print(f"- diameter: {nx.diameter(self)}")
        except Exception as ex:
            print(f"- diameter: <{ex}>")
        try:
            print(f"- average shortest path length: {nx.average_shortest_path_length(self)}")
        except Exception as ex:
            print(f"- average shortest path length: <{ex}>")
        print(f"- average degree: {net.get_average_degree(self)}")
        print(f"- degree assortativity: {nx.degree_assortativity_coefficient(self)}")
        print(f"- attribute assortativity ({self.class_attribute}): "
              f"{nx.attribute_assortativity_coefficient(self, self.class_attribute)}")
        print(f"- transitivity: {nx.transitivity(self)}")
        print(f"- average clustering: {nx.average_clustering(self)}")
        try:
            self.info_computed()
        except NotImplementedError as ex:
            print(f"Could not dynamically infer attributes: <{ex}>")

    def calculate_minimum_degree(self) -> int:
        """
        Returns the minimum degree of the graph.

        Returns
        -------
        int
            minimum degree
        """
        return net.get_min_degree(self)

    def calculate_fraction_of_minority(self) -> float:
        """
        Returns the fraction of minority nodes in the graph (based on class attribute).

        Returns
        -------
        float
            fraction of minority nodes
        """
        return net.get_minority_fraction(self)

    def calculate_edge_type_counts(self) -> Counter:
        """
        Returns the number of edges of each type, e.g., Mm (between majority and minority), mm, etc.

        Returns
        -------
        Counter
            counter of edges types
        """
        return net.get_edge_type_counts(self)

    def fit_powerlaw(self, metric: str) -> tuple[powerlaw.Fit, powerlaw.Fit]:
        """
        Fits a power law to the distribution given by 'metric' (the in- or out-degree of nodes in the graph).

        Parameters
        ----------
        metric: str
            metric to fit power law to

        Returns
        -------
        powerlaw.Fit
            power law fit of the majority class

        powerlaw.Fit
            power law fit of the minority class
        """

        fit_M, fit_m = fit_powerlaw_groups(self, metric)

        return fit_M, fit_m

    def calculate_powerlaw_exponents(self, metric: str) -> tuple[float, float]:
        """
        Returns the power law exponents for the ``metric`` distribution of the majority and minority class.

        Parameters
        ----------
        metric: str
            Metric to calculate the power law exponents for.

        Returns
        -------
        Tuple[float, float]
            power law exponents for the ``metric`` distribution of the majority and minority class

        Raises
        ------
            ValueError: Value of ``metric`` ∈ ['in_degree', 'out_degree'] if the graph is directed,
            otherwise it must be 'degree'.
        """
        metrics = ['in_degree', 'out_degree'] if self.is_directed() else ['degree']
        val.validate_values(metric, metrics)
        fit_M, fit_m = self.fit_powerlaw(metric=metric)
        pl_M = fit_M.power_law.alpha
        pl_m = fit_m.power_law.alpha
        return pl_M, pl_m

    ############################################################
    # Metadata
    ############################################################

    def compute_node_stats(self, metric: str, **kwargs) -> list[Union[int, float]]:
        """
        Returns the property of each node in the graph based on the metric.

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
            values = self.degree(self.node_list, **kwargs) if not self.is_directed() else None
        if metric == 'in_degree':
            values = self.in_degree(self.node_list, **kwargs) if self.is_directed() else None
        if metric == 'out_degree':
            values = self.out_degree(self.node_list, **kwargs) if self.is_directed() else None
        if metric == 'eigenvector':
            try:
                values = nx.eigenvector_centrality_numpy(self, **kwargs)
            except Exception as ex:
                try:
                    values = nx.eigenvector_centrality_numpy(self, max_iter=200, tol=1.0e-5)
                except Exception as ex:
                    warnings.warn(f"The eigenvector centrality could not be computed: {ex}")
                    values = None

        # dict of node -> value
        if metric == 'clustering':
            values = nx.clustering(self, self.node_list, **kwargs)
        if metric == 'betweenness':
            values = nx.betweenness_centrality(self, **kwargs)
        if metric == 'closeness':
            if isinstance(self, nx.DiGraph):
                values = nx.closeness_centrality(nx.DiGraph(self), **kwargs)
            else:
                values = nx.closeness_centrality(self, **kwargs)
        if metric == 'pagerank':
            values = nx.pagerank(self, **kwargs)

        return [values[n] for n in self.node_list] if values is not None else np.nan

    def get_node_metadata_as_dataframe(self, include_graph_metadata: bool = False, n_jobs: int = 1) -> pd.DataFrame:
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

        obj = {'node': self.node_list,
               'class_label': [const.MAJORITY_LABEL if self.get_class_label_by_node(
                   n) == self.get_majority_label() else const.MINORITY_LABEL for n in self.node_list],
               'real_label': [self.get_class_label_by_node(n) for n in self.node_list],
               'source': 'model' if 'empirical' not in self.graph else 'data'}

        # include graph metadata
        if include_graph_metadata:
            # n = self.number_of_nodes()
            new_cols = [c for c in self.graph.keys() if c not in ['class_attribute', 'class_values',
                                                                  'class_labels']]
            obj.update({c: self.graph[c] for c in new_cols})
            cols.extend(new_cols)

        # include metrics
        column_values = pqdm(const.VALID_METRICS, self.compute_node_stats, n_jobs=n_jobs)
        obj.update({col: values for col, values in zip(const.VALID_METRICS, column_values)})
        cols.extend(const.VALID_METRICS)

        # create dataframe
        df = pd.DataFrame(obj, columns=cols, index=self.node_list)
        df.name = self.model_name

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

    def makecopy(self) -> Union[nx.Graph, nx.DiGraph]:
        pass

    def copy(self) -> Union[nx.Graph, nx.DiGraph]:
        """
        Makes a copy of the current object.
        Returns
        -------
        netin.Graph
            copy of the current object
        """
        g = self.makecopy()
        g.init_graph(class_attribute=self.class_attribute,
                     class_values=self.class_values,
                     class_labels=self.class_labels)
        g.model_name = self.model_name

        g.graph.update(self.graph)
        g.add_nodes_from((n, d.copy()) for n, d in self._node.items())
        g.add_edges_from(
            (u, v, datadict.copy())
            for u, nbrs in self._adj.items()
            for v, datadict in nbrs.items()
        )
        g.node_list = self.node_list.copy()
        g.node_class_values = self.node_class_values.copy()
        g.n_m = self.n_m
        g.n_M = self.n_M
        return g


######################################################################################################################
# Static functions
######################################################################################################################

def fit_powerlaw_groups(g: Graph, metric: str) -> tuple[powerlaw.Fit, powerlaw.Fit]:
    """
    Fits a power law to the distribution given by 'metric' (the in- or out-degree of nodes in the graph).

    Parameters
    ----------
    g: Graph
        Graph to fit power law to

    metric: str
        metric to fit power law to

    Returns
    -------
    powerlaw.Fit
        power law fit of the majority class

    powerlaw.Fit
        power law fit of the minority class
    """

    def _get_value_fnc(g: Graph, metric: str) -> Iterable:
        if metric not in ['in_degree', 'out_degree', 'degree']:
            raise ValueError(f"`metric` must be either 'in_degree', 'out_degree' or 'degree', not {metric}")
        return g.in_degree if metric == 'in_degree' else g.out_degree if metric == 'out_degree' else g.degree

    vM = g.get_majority_value()
    fnc = _get_value_fnc(g, metric)

    dM = [d for n, d in fnc if g.node_class_values[n] == vM]
    dm = [d for n, d in fnc if g.node_class_values[n] != vM]

    fit_M = powerlaw.Fit(data=dM, discrete=True, xmin=min(dM), xmax=max(dM), verbose=False)
    fit_m = powerlaw.Fit(data=dm, discrete=True, xmin=min(dm), xmax=max(dm), verbose=False)
    return fit_M, fit_m


def convert_networkx_to_netin(g: Union[nx.Graph, nx.DiGraph], name: str, class_attribute: str) -> Graph:
    """
    Given a networkx graph, it creates a netin graph with the same structure and attributes.

    Parameters
    ----------
    g:  networkx
        Graph to convert

    name:  str
        name of the dataset

    class_attribute:  str
        name of the attribute that contains the class label

    Returns
    -------
    netin.Graph
        netin graph with the same structure and attributes
    """
    n = g.number_of_nodes()
    f_m = net.get_minority_fraction(g, class_attribute)
    seed = None

    if g.is_directed():
        d = nx.density(g)

        fit_M, fit_m = fit_powerlaw_groups(g, 'out_degree')
        plo_M = fit_M.power_law.alpha
        plo_m = fit_m.power_law.alpha
        # plo_M, plo_m = netin.calculate_out_degree_powerlaw_exponents(g)
        gn = netin.DiGraph(n=n, d=d, f_m=f_m, plo_M=plo_M, plo_m=plo_m, seed=seed)
    else:
        k = net.get_min_degree(g)
        gn = netin.UnDiGraph(n=n, k=k, f_m=f_m, seed=seed)

    gn.model_name = name
    gn.add_nodes_from(g.nodes(data=True))
    gn.add_edges_from(g.edges(data=True))

    counter = Counter([obj[class_attribute] for n, obj in g.nodes(data=True)])
    class_values, class_counts = zip(*counter.most_common())  # from M to m
    class_labels = ['female' if c == 1 else 'male' if c == 0 else 'unknown' for c in class_values]
    gn.initialize(class_attribute=class_attribute, class_values=class_values, class_labels=class_labels)

    gn.nodes = np.asarray(g.nodes())
    gn.node_class_values = {n: obj[class_attribute] for n, obj in g.nodes(data=True)}

    obj = {'model': gn.model_name,
           'e': g.number_of_edges()}

    gn.graph.update(obj)

    return gn
