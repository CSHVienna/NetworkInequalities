from typing import Union, Optional

import numpy as np

from .model import Model
from ..utils.constants import CLASS_ATTRIBUTE, MAJORITY_LABEL, MINORITY_LABEL
from ..graphs.binary_class_node_vector import BinaryClassNodeVector

class BinaryClassModel(Model):
    """An abstract class for models with binary class node values,
    grouping nodes into a single minority or majority group.

    Parameters
    ----------
    f_m : float
        The fraction of minority nodes.
    seed : Union[int, np.random.Generator], optional
        The random seed or generator, by default 1
    """
    f_m: float

    def __init__(
            self, *args,
            f_m: float,
            seed: Optional[Union[int, np.random.Generator]] = None,
            **kwargs):
        super().__init__(*args, seed=seed, **kwargs)
        self.f_m = f_m

    def _initialize_node_classes(self):
        """Initializes the node classes.
        If the attributes are present already (for instance, for pre-loaded graphs),
        they are kept and extended to the total number of nodes.
        Otherwise, a new binary class node vector is created.
        """
        if self.graph.has_node_class(CLASS_ATTRIBUTE):
            node_class_values_pre = self.graph.get_node_class(CLASS_ATTRIBUTE)
            assert isinstance(node_class_values_pre, BinaryClassNodeVector),\
            "The node class values must be binary"
            if len(node_class_values_pre) < self._n_nodes_total:
                ncv_post = BinaryClassNodeVector.from_fraction(
                    N=self._n_nodes_total,
                    f_m=self.f_m,
                    class_labels=[MAJORITY_LABEL, MINORITY_LABEL],
                    rng=self._rng)
                ncv_post[:len(node_class_values_pre)] =\
                node_class_values_pre.vals()
                self.graph.set_node_class(CLASS_ATTRIBUTE, ncv_post)
        else:
            self.graph.set_node_class(
                CLASS_ATTRIBUTE,
                BinaryClassNodeVector.from_fraction(
                    N=self._n_nodes_total,
                    f_m=self.f_m,
                    class_labels=[MAJORITY_LABEL, MINORITY_LABEL],
                    rng=self._rng))
