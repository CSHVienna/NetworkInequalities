import gc
from collections import defaultdict
from typing import \
    Dict

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
from scipy.sparse import triu
from scipy.special import gammaln
from sklearn.preprocessing import normalize

from ...base_class import BaseClass
from ...graphs.graph import Graph
from ...utils import io


class JanusGraph(BaseClass):

    def __init__(self, graph: Graph, **attr):
        BaseClass.__init__(self, **attr)
        self.graph = graph  # netin.Graph
        self.adj_matrix: lil_matrix = None
        self.adj_matrix_clean: lil_matrix = None

    def init_data(self, is_global: bool = True):
        self.adj_matrix = self.graph.get_adjacency_matrix().tolil() * 100
        if is_global:
            if self.is_directed():
                # flatten 1 x n^2
                self.adj_matrix_clean = self.adj_matrix.reshape(1, -1)
            else:
                # just one side (eg. upper diagonal)
                self.adj_matrix_clean = triu(self.adj_matrix, k=1)
        else:
            self.adj_matrix_clean = self.adj_matrix


    def is_directed(self):
        return self.graph.is_directed()

    def number_of_nodes(self):
        return self.graph.number_of_nodes()

    def get_node_classes(self):
        return self.graph.get_node_classes()


class Hypothesis(BaseClass):

    def __init__(self, name: str, belief_matrix: lil_matrix, graph: JanusGraph, is_global: bool, **attr):
        BaseClass.__init__(self, **attr)
        self.name = name
        self.graph = graph
        self.is_global = is_global
        self.belief_matrix = belief_matrix
        self.belief_matrix_normalized: lil_matrix = None
        self._init_hypothesis()

    def _init_hypothesis(self):
        # set the correct dimensions
        if self.is_global:
            if self.graph.is_directed():
                # flatten 1 x n^2
                belief_matrix = self.belief_matrix.reshape(1, -1)
            else:
                # just one side (eg. upper diagonal)
                belief_matrix = triu(self.belief_matrix, k=1)
        else:
            belief_matrix = self.belief_matrix

        # normalize
        self.belief_matrix_normalized = normalize(belief_matrix, axis=1, norm='l1', copy=True)

    def elicit_prior(self, k: int) -> lil_matrix:
        n = self.graph.number_of_nodes()
        kappa = n * (n if self.is_global else 1.0) * k

        if k in [0., 0.1]:
            prior = csr_matrix(self.belief_matrix_normalized.shape, dtype=np.float64)
        else:
            prior = self.belief_matrix_normalized.copy() * kappa

            # rows only 0 --> k
            norma = prior.sum(axis=1)
            n_zeros, _ = np.where(norma == 0)
            prior[n_zeros, :] = k

        return prior

    def compute_evidence(self, prior: lil_matrix, k: int) -> float:
        """
        Computes the Categorical Dirichlet evidence
        Parameters
        ----------
        prior
        k

        Returns
        -------
        float
            Value of the log of the absolute value of gamma
        """
        n = self.graph.number_of_nodes()
        proto_prior = 1.0 + (k if prior.size == 0 else 0.)
        uniform = n * (n if self.is_global else 1.0) * proto_prior
        evidence = 0
        evidence += gammaln(prior.sum(axis=1) + uniform).sum()
        evidence -= gammaln(self.graph.adj_matrix_clean.sum(axis=1) + prior.sum(axis=1) + uniform).sum()

        self.log('shape graph.data: {} | size: {}'.format(self.graph.adj_matrix_clean.shape,
                                                          self.graph.adj_matrix_clean.size))
        self.log('shape prior: {} | size:{}'.format(prior.shape, prior.size))

        evidence += gammaln((self.graph.adj_matrix_clean + prior).data + proto_prior).sum()
        evidence -= gammaln(prior.data + proto_prior).sum() + (
                (self.graph.adj_matrix_clean.size - prior.size) * gammaln(proto_prior))
        ### the uniform is added since it is the starting point for the first value of k
        ### the last negative sum includes (graph.size - prior.size) * uniform to include all empty cells
        return evidence


class Janus(BaseClass):

    def __init__(self, graph: Graph, is_global: bool = True, k_max: int = 10, k_log_scale: bool = True, **attr):
        BaseClass.__init__(self, **attr)
        self.graph: JanusGraph = JanusGraph(graph, **attr)  # graph (from class Graph)
        self.graph.init_data(is_global)
        self.is_global = is_global  # kind of testing: global or local
        self.k_max = k_max  # maximun value of weighting factor
        self.k_log_scale = k_log_scale  # whether the weighting factors should be log scale or not
        self.weighting_factors: np.ndarray = None
        # self.hypotheses: Set[Hypothesis] = set()  # Belief matrices
        self.evidences: Dict[str, Dict[float, float]] = {}  # Evidence values for each hypothesis and weighting factor k
        self._init_weighting_factors()

    ### DEFAULT HYPOTHESES
    def get_uniform_hypothesis(self) -> Hypothesis:
        n = self.graph.number_of_nodes()
        h = Hypothesis(name='uniform', belief_matrix=lil_matrix((n, n)), graph=self.graph, is_global=self.is_global)
        return h

    def get_self_loop_hypothesis(self) -> Hypothesis:
        n = self.graph.number_of_nodes()
        diagonal_matrix = lil_matrix((n, n))
        diagonal_matrix.setdiag(1.)
        h = Hypothesis(name='self_loop', belief_matrix=diagonal_matrix.tolil(), graph=self.graph,
                       is_global=self.is_global)
        return h

    def get_data_hypothesis(self) -> Hypothesis:
        h = Hypothesis(name='data', belief_matrix=self.graph.adj_matrix.tolil(), graph=self.graph,
                       is_global=self.is_global)
        return h

    def get_model_hypothesis(self, model_gen, **args):
        g = model_gen(**args)
        g.simulate()
        h = Hypothesis(name=g.SHORT,
                       belief_matrix=g.graph.get_adjacency_matrix().tolil(),
                       graph=self.graph,
                       is_global=self.is_global)
        return h

    def get_homophily_hypothesis(self, h_m: float = 1.0, h_M: float = 1.0):
        assert h_m >= 0 and h_m <= 1.0 and h_M >= 0.0 and h_M <= 1.0
        name = f"hm{h_m}_hM{h_M}"

        indices_m = np.where(self.graph.get_node_classes()['minority'].get_values() == 1)[0]
        indices_M = np.where(self.graph.get_node_classes()['minority'].get_values() == 0)[0]

        # Size of the matrix
        size = max(max(indices_m), max(indices_M)) + 1

        # Create a lil_matrix of the required size
        belief_matrix = lil_matrix((size, size))

        # Set values for elements in indices_m and indices_M
        belief_matrix[np.ix_(indices_m, indices_m)] = h_m
        belief_matrix[np.ix_(indices_M, indices_M)] = h_M

        h = Hypothesis(name=name,
                       belief_matrix=belief_matrix,
                       graph=self.graph,
                       is_global=self.is_global)
        return h

    def model_fitting(self, model_gen, n_iter=10, **args):
        verbose = self.get_verbose()
        self.set_verbose(False)

        evidences = {}
        for i in range(n_iter):
            h = self.get_model_hypothesis(model_gen, **args)
            e = self.generate_evidences(h)
            evidences[f"{h.name}_{i}"] = e

        # average
        sums = defaultdict(int)
        counts = defaultdict(int)

        # Iterate over each inner dictionary and sum the values for each key
        for inner_dict in evidences.values():
            for key, value in inner_dict.items():
                sums[key] += value
                counts[key] += 1

        # Compute the average for each key
        averages = {key: sums[key] / counts[key] for key in sums}
        evidences = {model_gen.SHORT: averages}
        self.set_verbose(verbose)
        return evidences

    ### EVIDENCE
    def _init_weighting_factors(self):
        '''
        Initializes the weighting factors for the prior elicitation
        Parameters
        ----------
        k_max
        k_log_scale

        Returns
        -------

        '''
        if self.k_log_scale:
            self.weighting_factors = np.sort(np.unique(np.append(np.logspace(-1, self.k_max, self.k_max + 2), 1)))
        else:
            self.weighting_factors = np.arange(0, self.k_max + 1, 1).astype(float)

    def generate_evidences(self, h: Hypothesis):
        self.log('===== CALCULATING EVIDENCES =====')
        self.log(f'::: Hypothesis: {h.name} :::')
        evidences = {}

        for k in self.weighting_factors:
            prior = h.elicit_prior(k)
            e = h.compute_evidence(prior, k)
            evidences[k] = e
            self.log(f"h:{h.name:20s}, k={k:10.2f}, e={e:10.2f}")
            del (prior)
            del (e)
            gc.collect()

        return evidences

    def add_evidences(self, name: str, evidences: Dict[float, float]):
        self.evidences[name] = evidences

    def save_evidences(self, output_dir: str):
        assert output_dir != ''
        assert io.validate_dir(output_dir)
        fn = io.path_join(output_dir, 'evidences.json')
        io.save_dict_to_file(self.evidences, fn)

    def plot_evidences(self, bayes_factors: bool = False, output_dir: str = None, **kwargs):
        import matplotlib.pyplot as plt
        import operator

        assert (bayes_factors and 'uniform' in self.evidences) or (not bayes_factors)

        fig = plt.figure(figsize=kwargs.get('figsize', (10, 5)))
        ax = fig.add_subplot(111)
        fig.canvas.draw()

        n = self.graph.number_of_nodes()
        for hypothesis_name, evidences_obj in self.evidences.items():
            sorted_evidences_by_k = sorted(evidences_obj.items(), key=operator.itemgetter(0), reverse=False)

            yy = [e[1] - (self.evidences['uniform'][e[0]] if bayes_factors else 0) for e in sorted_evidences_by_k]
            xx = [float(e[0]) * (n * (n if self.is_global else 1)) for e in sorted_evidences_by_k]

            if self.k_log_scale:
                plot = ax.semilogx
            else:
                plot = ax.plot

            plot(xx, yy, label=hypothesis_name)  # marker, color

        k_rank = max([e[0] for e in sorted_evidences_by_k])
        ax.set_xlabel("concentration parameter $\kappa$")
        ax.set_ylabel("log(evidence)" if not bayes_factors else "log(Bayes factor)")
        plt.grid(False)
        ax.xaxis.grid(True)

        # if self.is_global:
        #     ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))

        ### Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        handles, labels = ax.get_legend_handles_labels()
        tmp = {hypothesis_name: evidence
               for hypothesis_name, obj in self.evidences.items()
               for k, evidence in obj.items() if k == k_rank}

        t = [(l, h, tmp[l]) for l, h in zip(labels, handles)]
        labels, handles, evidences = zip(*sorted(t, key=lambda t: t[2], reverse=True))
        legend = ax.legend(handles, labels, loc='upper center',
                           bbox_to_anchor=(kwargs.get('bboxx', 1.18),
                                           kwargs.get('bboxy', 1.0)),
                           fontsize=kwargs.get('fontsize', 13),
                           ncol=kwargs.get('ncol', 1))  # inside
        ax.grid('on')

        if output_dir is not None:
            io.validate_dir(output_dir)
            plt.savefig(io.path_join(output_dir, f"evidences_{'global' if self.is_global else 'local'}.pdf"),
                        bbox_extra_artists=(legend,),
                        bbox_inches='tight',
                        dpi=kwargs.get('dpi', 1200))

        plt.show()
        plt.close()
