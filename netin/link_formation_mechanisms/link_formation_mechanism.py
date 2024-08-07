from abc import abstractmethod

import numpy as np

class LinkFormationMechanism:
    @abstractmethod
    def get_target_probabilities(self, source: int) -> np.ndarray:
        raise NotImplementedError
