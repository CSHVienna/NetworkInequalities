from collections import defaultdict
from typing import \
    Dict

import numpy as np
from scipy.sparse import lil_matrix

from .hypothesis import Hypothesis
from .janus import Janus
from ...graphs.graph import Graph
from ...models import HomophilyModel
from ...models import PAHModel
from ...models import PAModel


class JanusModelFitting(Janus):
    def __init__(self, graph: Graph, is_global: bool = True, k_max: int = 10, k_log_scale: bool = True, **attr):
        super().__init__(graph, is_global, k_max, k_log_scale, **attr)

    #######################################
    ### MODEL HYPOTHESIS / BELIEF-BASED ###
    #######################################

    def model_fitting_belief_based(self, model_gen, first_mover_bias: bool = False) -> Dict[
        str, Dict[float, float]]:
        verbose = self.get_verbose()
        self.set_verbose(False)

        if model_gen.SHORT == HomophilyModel.SHORT:
            best_evidence = self._H_fitting(first_mover_bias)
        elif model_gen.SHORT == PAModel.SHORT:
            best_evidence = self._PA_fitting(first_mover_bias)
        elif model_gen.SHORT == PAHModel.SHORT:
            best_evidence = self._PAH_fitting(first_mover_bias)
        else:
            raise NotImplementedError('Model not implemented')

        self.set_verbose(verbose)
        return best_evidence

    ### Homophily

    def _H_fitting(self, first_mover_bias: bool = False) -> Dict[str, Dict[float, float]]:
        evidences = {}
        for h_mm in np.linspace(0., 1., 10):
            for h_MM in np.linspace(0., 1., 10):
                h = self.get_H_hypothesis(round(h_mm, 1), round(h_MM, 1), first_mover_bias)
                e = self.generate_evidences(h)
                evidences[h.name] = e

        return self._get_best_evidence(evidences)

    def get_H_hypothesis(self, h_mm: float = 1.0, h_MM: float = 1.0, first_mover_bias: bool = False) -> Hypothesis:
        assert h_mm >= 0 and h_mm <= 1.0 and h_MM >= 0.0 and h_MM <= 1.0
        name = f"H_hmm{h_mm}_hMM{h_MM}"

        indices_m = np.where(self.graph.get_node_class().get_minority_mask())[0]
        indices_M = np.where(self.graph.get_node_class().get_majority_mask())[0]

        # Size of the matrix
        size = max(max(indices_m), max(indices_M)) + 1

        # Create a lil_matrix of the required size
        belief_matrix = lil_matrix((size, size))

        # Bias due to first_mover_advantage
        bias_m = 1.0 if not first_mover_bias else 1 / (indices_m + 1)
        bias_M = 1.0 if not first_mover_bias else 1 / (indices_M + 1)

        # Set values for elements in indices_m and indices_M
        belief_matrix[np.ix_(indices_m, indices_m)] = h_mm * bias_m
        belief_matrix[np.ix_(indices_M, indices_M)] = h_MM * bias_M
        belief_matrix[np.ix_(indices_m, indices_M)] = (1 - h_mm) * bias_M
        belief_matrix[np.ix_(indices_M, indices_m)] = (1 - h_MM) * bias_m

        # setting diagonal to 0
        belief_matrix.setdiag(np.zeros(belief_matrix.shape[0]))

        h = Hypothesis(name=name,
                       belief_matrix=belief_matrix,
                       graph=self.graph,
                       is_global=self.is_global)
        return h

    ### Preferential Attachment

    def _PA_fitting(self, first_mover_bias: bool = False) -> Dict[str, Dict[float, float]]:
        evidences = {}
        h = self.get_PA_hypothesis(first_mover_bias)
        e = self.generate_evidences(h)
        evidences[h.name] = e
        return evidences

    def get_PA_hypothesis(self, first_mover_bias: bool = False) -> Hypothesis:
        name = f"PA"

        degree = np.array(self.graph.adj_matrix.sum(axis=0))  # colum-wise

        # Size of the matrix
        indices = range(0, self.graph.number_of_nodes())
        size = max(max(indices), max(indices)) + 1

        # Create a lil_matrix of the required size
        belief_matrix = lil_matrix((size, size))

        # Bias due to first_mover_advantage
        bias = 1.0 if not first_mover_bias else (1.0 / np.arange(1, self.graph.number_of_nodes() + 1, 1)).reshape(1,
                                                                                                                  -1)

        # Set values for elements in indices_m and indices_M
        belief_matrix[np.ix_(indices, indices)] = degree[0, indices] * bias

        # setting diagonal to 0
        belief_matrix.setdiag(np.zeros(belief_matrix.shape[0]))

        h = Hypothesis(name=name,
                       belief_matrix=belief_matrix,
                       graph=self.graph,
                       is_global=self.is_global)
        return h

    ### Preferential Attachment + Homophily

    def _PAH_fitting(self, first_mover_bias: bool = False) -> Dict[str, Dict[float, float]]:
        evidences = {}
        for h_mm in np.linspace(0, 1, 10):
            for h_MM in np.linspace(0, 1, 10):
                h = self.get_PAH_hypothesis(round(h_mm, 1), round(h_MM, 1), first_mover_bias)
                e = self.generate_evidences(h)
                evidences[h.name] = e

        return self._get_best_evidence(evidences)

    def get_PAH_hypothesis(self, h_mm: float = 1.0, h_MM: float = 1.0, first_mover_bias: bool = False) -> Hypothesis:
        assert h_mm >= 0 and h_mm <= 1.0 and h_MM >= 0.0 and h_MM <= 1.0
        name = f"PAH_hmm{h_mm}_hMM{h_MM}"

        indices_m = np.where(self.graph.get_node_class().get_minority_mask())[0]
        indices_M = np.where(self.graph.get_node_class().get_majority_mask())[0]
        degree = np.array(self.graph.adj_matrix.sum(axis=0))  # colum-wise

        # Size of the matrix
        size = max(max(indices_m), max(indices_M)) + 1

        # Create a lil_matrix of the required size
        belief_matrix = lil_matrix((size, size))

        # Bias due to first_mover_advantage
        bias_m = 1.0 if not first_mover_bias else (1 / (indices_m + 1)).reshape(1, -1)
        bias_M = 1.0 if not first_mover_bias else (1 / (indices_M + 1)).reshape(1, -1)

        # Set values for elements in indices_m and indices_M
        belief_matrix[np.ix_(indices_m, indices_m)] = h_mm * degree[0, indices_m] * bias_m
        belief_matrix[np.ix_(indices_M, indices_M)] = h_MM * degree[0, indices_M] * bias_M
        belief_matrix[np.ix_(indices_m, indices_M)] = (1 - h_mm) * degree[0, indices_M] * bias_M
        belief_matrix[np.ix_(indices_M, indices_m)] = (1 - h_MM) * degree[0, indices_m] * bias_m

        # setting diagonal to 0
        belief_matrix.setdiag(np.zeros(belief_matrix.shape[0]))

        h = Hypothesis(name=name,
                       belief_matrix=belief_matrix,
                       graph=self.graph,
                       is_global=self.is_global)
        return h

    #######################################
    ### MODEL HYPOTHESIS / MODEL-BASED ###
    #######################################

    def get_model_hypothesis(self, model_gen, **args) -> Hypothesis:
        g = model_gen(**args)
        g.simulate()
        h = Hypothesis(name=g.SHORT,
                       belief_matrix=g.graph.get_adjacency_matrix().tolil(),
                       graph=self.graph,
                       is_global=self.is_global)
        return h

    def model_fitting(self, model_gen, n_iter=10, **args) -> Dict[str, Dict[float, float]]:
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
