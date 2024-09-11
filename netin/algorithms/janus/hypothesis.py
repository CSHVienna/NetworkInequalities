import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
from scipy.sparse import triu
from scipy.special import gammaln
from sklearn.preprocessing import normalize

from .janusgraph import JanusGraph
from ...base_class import BaseClass


class Hypothesis(BaseClass):
    """
    Hypothesis class that represents a belief of edge formation of a given graph
    """

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
        """
        Elicits the prior for the given belief matrix
        Parameters
        ----------
        k

        Returns
        -------
        lil_matrix
            Prior matrix

        """
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
