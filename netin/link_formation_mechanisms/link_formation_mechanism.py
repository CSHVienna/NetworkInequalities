from abc import abstractmethod

import numpy as np

from ..base_class import BaseClass

class LinkFormationMechanism(BaseClass):
    @abstractmethod
    def get_target_probabilities(self, source: int) -> np.ndarray:
        raise NotImplementedError
