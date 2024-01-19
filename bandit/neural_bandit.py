import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.neural_network import MLPRegressor
import random

class NeuralBandit:
    def __init__(self, *args, **kwargs):
        self.model = MLPRegressor(*args, **kwargs)

    def fit(self, state: dict[str, str], action: str, reward: float):
        context = self.preprocess(state, action)
        X = np.array(context).reshape(1, -1)
        y = np.array(reward).ravel()
        self.model.partial_fit(X, y)

    def predict(self, state: dict[str, str], actions: list[str]) -> str:
        try:
            rewards = self.model.predict(
                [(context := self.preprocess(state, action)) for action in actions]
            )
            action = self.policy(rewards)
            return actions[action]
        except NotFittedError:
            return np.random.choice(actions)

    def policy(rewards: list[float]):
        raise NotImplemented("policy not implemented")

    def preprocess(self, state: dict[str, str], action: str):
        raise NotImplemented("preprocess not implemented")


class EpsilonGreedyNeuralBandit(NeuralBandit):
    def __init__(self, epsilon = 0.9, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon # Default to 0.9, which means 10% exploration, 90% exploitation.
        self.__name__ = f'{self.__class__.__name__}_{epsilon}'

    def policy(self, rewards: list[float]) -> int:
        e = np.random.uniform(0, 1)
        if e < self.epsilon:
            return np.argmax(rewards)
        else:
            return np.random.choice(range(len(rewards)))


class SoftmaxNeuralBandit(NeuralBandit):
    def __init__(self, temperature = 0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature
        self.__name__ = f'{self.__class__.__name__}_{temperature}'

    def policy(self, rewards: list[float]):
        probabilities = softmax(rewards, temperature=self.temperature)
        return np.random.choice(range(len(rewards)), p=probabilities)


class NeuralPerArmBandit:
    def __init__(self, n_actions: int, *args, **kwargs):
        self.models = [MLPRegressor(*args, **kwargs) for _ in range(n_actions)]
        
    def fit(self, state: dict[str, str], action: int, reward: float):
        context = self.preprocess(state)
        X = np.array(context).reshape(1, -1)
        y = np.array(reward).ravel()
        self.models[action].partial_fit(X, y)

    def predict(self, state: dict[str, str], actions: list[str]):
        try:
            rewards = np.array([model.predict(self.preprocess(state).reshape(1, -1)) for model in self.models])
            action = self.policy(rewards.ravel())
            return actions[action]
        except NotFittedError:
            return np.random.choice(actions)

    def policy(rewards):
        raise NotImplemented("policy not implemented")

    def preprocess(self, state: dict[str, str]):
        raise NotImplemented("preprocess not implemented")
        
class EpsilonGreedyNeuralPerArmBandit(NeuralPerArmBandit):
    def __init__(self, n_actions: int, epsilon = 0.9, *args, **kwargs):
        super().__init__(n_actions, *args, **kwargs)
        self.epsilon = epsilon
        self.__name__ = f'{self.__class__.__name__}_{epsilon}'

    def policy(self, rewards: list[float]) -> int:
        e = np.random.uniform(0, 1)
        if e < self.epsilon:
            return np.argmax(rewards)
        else:
            return np.random.choice(range(len(rewards)))

class SoftmaxNeuralPerArmBandit(NeuralPerArmBandit):
    def __init__(self, n_actions: int, temperature = 0.2, *args, **kwargs):
        super().__init__(n_actions, *args, **kwargs)
        self.temperature = temperature
        self.__name__ = f'{self.__class__.__name__}_{temperature}'

    def policy(self, rewards: list[float]):
        probabilities = softmax(rewards, temperature=self.temperature)
        return np.random.choice(range(len(rewards)), p=probabilities)


def softmax(lst, temperature=1.0):
    lst = np.array(lst) / temperature
    exps = np.exp(lst)
    return exps / np.sum(exps)
