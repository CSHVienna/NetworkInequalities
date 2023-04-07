import numpy as np

from netin.utils import constants as const
from netin.utils import validator as val


class Metadata(object):

    def __init__(self, n: int, k: int, f_m: float, h_MM: float, h_mm: float, tc: float, seed: object):
        """

        Parameters
        ----------
        n: number of nodes
        k: minimum degree of nodes
        f_m: fraction of minorities
        h_MM: homophily within the majority group
        h_mm: homphily within the minority group
        tc: triadic closure probability
        seed: seed for random number generator
        """
        self.n = n
        self.k = k
        self.f_m = f_m
        self.h_MM = h_MM
        self.h_mm = h_mm
        self.tc = tc
        self.seed = seed if seed is not None else np.random.randint(0, 2 ** 32)
        self.n_m = 0
        self.n_M = 0
        self.model_name = None
        self.class_attribute = None
        self.class_values = None
        self.class_labels = None
        self.mixing_matrix = None

    ############################################################
    # Graph metadata
    ############################################################

    def _infer_model_name(self):
        """
        Infers the name of the model.
        """
        self.model_name = None
        if self.tc in const.NO_TRIADIC_CLOSURE and self.h_mm in const.NO_HOMOPHILY and self.h_MM in const.NO_HOMOPHILY:
            self.model_name = 'PA'
        if self.tc in const.NO_TRIADIC_CLOSURE and self.h_mm not in const.NO_HOMOPHILY and self.h_MM not in const.NO_HOMOPHILY:
            self.model_name = 'PAH'
        if self.tc not in const.NO_TRIADIC_CLOSURE and self.h_mm in const.NO_HOMOPHILY and self.h_MM in const.NO_HOMOPHILY:
            self.model_name = 'PATC'
        if self.tc not in const.NO_TRIADIC_CLOSURE and self.h_mm not in const.NO_HOMOPHILY and self.h_MM not in const.NO_HOMOPHILY:
            self.model_name = 'PATCH'
        if self.model_name is None:
            raise ValueError('Unknown model name.')

    def _set_class_info(self, class_attribute: str = 'm', class_values=None, class_labels=None):
        if class_labels is None:
            class_labels = [const.MAJORITY_LABEL, const.MINORITY_LABEL]
        if class_values is None:
            class_values = [0, 1]
        self.class_attribute = class_attribute
        self.class_values = class_values
        self.class_labels = class_labels

    def _validate_parameters(self):
        """
        Validates the parameters of the undigraph.
        """
        val.validate_int(self.n, minimum=2)
        val.validate_int(self.k, minimum=1)
        val.validate_float(self.f_m, minimum=1 / self.n, maximum=(self.n - 1) / self.n)
        val.validate_float(self.h_MM, minimum=0., maximum=1., allow_none=True)
        val.validate_float(self.h_mm, minimum=0., maximum=1., allow_none=True)
        val.validate_float(self.tc, minimum=0., maximum=1., allow_none=True)

    def _init_nodes(self):
        nodes = np.arange(self.n)
        self.n_M = int(round(self.n * (1 - self.f_m)))
        self.n_m = self.n - self.n_M
        minorities = np.random.choice(nodes, self.n_m, replace=False)
        labels = {n: int(n in minorities) for n in nodes}
        return nodes, labels

    def get_metadata_as_dict(self) -> dict:
        """
        Returns metadata for a undigraph.
        """
        obj = {'name': self.get_model_name(),
               'class_attribute': self.get_class_attribute(),
               'class_values': self.get_class_values(),
               'class_labels': self.get_class_labels(),
               'n': self.n,
               'k': self.k,
               'f_m': self.f_m,
               'h_MM': self.h_MM,
               'h_mm': self.h_mm,
               'tc': self.tc,
               'seed': self.seed}
        return obj

    ############################################################
    # Homophily
    ############################################################

    def if_homophily_in_model(self) -> bool:
        return self.get_model_name() in const.HOMOPHILY_MODELS

    def _init_mixing_matrix(self):
        if self.if_homophily_in_model():
            self.h_MM = val.validate_homophily(self.h_MM)
            self.h_mm = val.validate_homophily(self.h_mm)
            self.mixing_matrix = np.array([[self.h_MM, 1 - self.h_MM], [1 - self.h_mm, self.h_mm]])

    ############################################################
    # Getters
    ############################################################

    def get_model_name(self):
        return self.model_name

    def get_class_attribute(self):
        return self.class_attribute

    def get_class_values(self):
        return self.class_values

    def get_class_labels(self):
        return self.class_labels




