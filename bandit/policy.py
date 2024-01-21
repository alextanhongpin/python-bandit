import numpy as np
from abc import ABC, abstractmethod


class Policy(ABC):
    @abstractmethod
    def policy(self, vals: list[float]) -> int:
        pass

    def __call__(self, vals: list[float]):
        return self.policy(vals)


class Softmax(Policy):
    def __init__(self, tau=0.2, rng=np.random.RandomState(None)):
        self.tau = tau
        self.rng = rng

    def policy(self, vals: list[float]) -> int:
        vals = np.array(vals)
        probs = self.softmax(vals)
        return self.rng.choice(range(len(vals)), p=probs)

    def softmax(self, vals) -> list[float]:
        vals = np.array(vals) / self.tau
        exps = np.exp(vals)
        return exps / np.sum(exps)


class EGreedy(Policy):
    def __init__(self, epsilon=0.1, rng=np.random.RandomState(None)):
        self.epsilon = epsilon
        self.rng = rng

    def policy(self, vals: list[float]) -> int:
        vals = np.array(vals)
        max_val = max(vals)
        if self.explore():
            choices = np.argwhere(vals != max_val).flatten()
            # All same values.
            if len(choices) != 0:
                return self.rng.choice(choices)
        return self.rng.choice(np.argwhere(vals == max_val).flatten())

    def explore(self) -> bool:
        return self.rng.random() < self.epsilon


def get_policy(policy: str | Policy) -> Policy:
    if isinstance(policy, Policy):
        return policy
    if policy == "softmax":
        return Softmax()
    if policy == "egreedy":
        return EGreedy()
    raise ValueError(f"Unknown policy: {policy}")
