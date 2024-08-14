from abc import abstractmethod

from ..base_class import BaseClass
from ..graphs.node_vector import NodeVector

class Filter(BaseClass):
    @abstractmethod
    def get_target_mask(self, source: int) -> NodeVector:
        raise NotImplementedError
