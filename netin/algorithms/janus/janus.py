import gc
from typing import \
    Dict

import numpy as np
from scipy.sparse import lil_matrix

from .hypothesis import Hypothesis
from .janusgraph import JanusGraph
from ...base_class import BaseClass
from ...graphs.graph import Graph
from ...utils import io


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

    ##########################
    ### DEFAULT HYPOTHESES ###
    ##########################

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

    ########################
    ### EVIDENCE         ###
    ########################

    def _get_best_evidence(self, evidences: Dict[str, Dict[float, float]]) -> Dict[str, Dict[float, float]]:
        max_model = max(evidences, key=lambda k: sum(evidences[k].values()))
        return {max_model: evidences[max_model]}

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

    def update_evidences(self, evidences: Dict[str, Dict[float, float]]):
        self.evidences.update(evidences)

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

        if self.is_global:
            ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))

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
