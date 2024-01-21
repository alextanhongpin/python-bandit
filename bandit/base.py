from abc import ABC, abstractmethod
import numpy as np
from itertools import product


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


def extract_features(state: dict, action: int) -> np.ndarray:
    """perform feature interactions, similar to how vowpal wabbit does it.
    We create additional features which are the features in the (U)ser namespace and (A)ction
    namespaces multiplied together.
    This allows us to learn the interaction between when certain actions are good in certain times of days and for particular users.
    If we didn’t do that, the learning wouldn’t really work.
    We can see that in action below.
    """
    features = []
    for kvs in product(state.items(), [("action", action)]):
        features.append("^".join([f"{k}:{v}" for k, v in kvs]))
    return np.array(features)
