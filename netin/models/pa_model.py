from typing import Union, Optional

import numpy as np

from .barabasi_albert_model import BarabasiAlbertModel
from .binary_class_model import BinaryClassModel


class PAModel(BarabasiAlbertModel, BinaryClassModel):
    """The PAModel is an extension of the BarabasiAlbertModel where besides new nodes joining existing nodes with a
    probability proportional to the degree of the existing nodes, nodes are assign a binary label (see [Karimi2018]_).
    """
    SHORT = "PA"

    def __init__(
            self, *args,
            n: int, f_m: float, m: int,
            seed: Optional[Union[int, np.random.Generator]] = None,
            **kwargs):
        super(BarabasiAlbertModel, self).__init__(*args, n=n, m=m, f_m=f_m, seed=seed, **kwargs)

    def _initialize_node_classes(self):
        BinaryClassModel._initialize_node_classes(self)
