from abc import abstractmethod

from ..base_class import BaseClass
from ..graphs.node_attributes import NodeAttributes

class Filter(BaseClass):
    @abstractmethod
    def get_target_mask(self, source: int) -> NodeAttributes:
        raise NotImplementedError
