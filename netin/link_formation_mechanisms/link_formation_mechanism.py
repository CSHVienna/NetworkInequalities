from abc import abstractmethod

import numpy as np

from ..base_class import BaseClass
from ..graphs.node_attributes import NodeAttributes

class LinkFormationMechanism(BaseClass):
    @abstractmethod
    def get_target_probabilities(self, source: int) -> NodeAttributes:
        raise NotImplementedError
