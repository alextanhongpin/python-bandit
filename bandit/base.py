from abc import ABC, abstractmethod
import numpy as np


class BaseBandit(ABC):
    def __init__(self, n_arms):
        self.n_arms = n_arms

    @abstractmethod
    def update(self, state: dict, action: int, reward: int):
        """update the model with the new reward"""
        pass

    @abstractmethod
    def pull(self, state: dict) -> np.ndarray:
        """returns the reward for each arm"""
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(n_arms={self.n_arms})"
