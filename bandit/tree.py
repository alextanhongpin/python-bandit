import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.tree import DecisionTreeRegressor


class TreeBandit:
    def __init__(self, *args, **kwargs):
        self.model = DecisionTreeRegressor(*args, **kwargs)
        self.action_history = []
        self.rewards = []

    def fit(self, state: dict[str, str], action: str, reward: float):
        context = self.preprocess(state, action)
        X = np.array(context)
        y = reward
        self.action_history.append(X)
        self.rewards.append(y)
        if len(np.unique(self.rewards)) > 1 and len(self.rewards) % 20 == 0:
            self.model.fit(self.action_history, self.rewards)

    def predict(self, state: dict[str, str], actions: list[str]) -> str:
        try:
            rewards = self.model.predict(
                [self.preprocess(state, action) for action in actions]
            )
            action = self.policy(rewards)
            return actions[action]
        except NotFittedError:
            return np.random.choice(actions)

    def policy(self, rewards: list[float]):
        raise NotImplementedError("policy not implemented")

    def preprocess(self, state: dict[str, str], action: str):
        raise NotImplementedError("preprocess not implemented")


class EpsilonGreedyTreeBandit(TreeBandit):
    def __init__(self, epsilon=0.9, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Default to 0.9, which means 10% exploration, 90%
        # exploitation.
        self.epsilon = epsilon
        self.__name__ = f"{self.__class__.__name__}_{epsilon}"

    def policy(self, rewards: list[float]) -> int:
        e = np.random.uniform(0, 1)
        if e < self.epsilon:
            return np.argmax(rewards)
        else:
            return np.random.choice(range(len(rewards)))


class SoftmaxTreeBandit(TreeBandit):
    def __init__(self, temperature=0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature
        self.__name__ = f"{self.__class__.__name__}_{temperature}"

    def policy(self, rewards: list[float]):
        probabilities = softmax(rewards, temperature=self.temperature)
        return np.random.choice(range(len(rewards)), p=probabilities)


class TreePerArmBandit:
    def __init__(self, n_actions: int, *args, **kwargs):
        self.models = [DecisionTreeRegressor() for _ in range(n_actions)]
        self.action_history = [[] for _ in range(n_actions)]
        self.rewards = [[] for _ in range(n_actions)]
        self.n_actions = n_actions

    def fit(self, state: dict[str, str], action: int, reward: float):
        context = self.preprocess(state)
        X = np.array(context).reshape(1, -1)
        y = reward
        self.action_history[action].append(X)
        self.rewards[action].append(y)
        _, n_features = X.shape
        # Only train if we have at least 2 targets.
        # We should also limit the training per timestep, e.g. every N-iterations etc.
        if len(np.unique(self.rewards[action])) >= 2:
            X = np.array(self.action_history[action]).reshape(-1, n_features)
            y = np.array(self.rewards[action])
            self.models[action].fit(X, y)

    def predict(self, state: dict[str, str], actions: list[str]) -> str:
        X = self.preprocess(state).reshape(1, -1)
        rewards = np.array([self._predict(model, X) for model in self.models])
        action = self.policy(rewards)
        return actions[action]

    def _predict(self, model, X):
        try:
            return model.predict(X)[0]
        except NotFittedError:
            return 0  # Make this selected

    def policy(self, rewards):
        raise NotImplementedError("policy not implemented")

    def preprocess(self, state: dict[str, str]):
        raise NotImplementedError("preprocess not implemented")


class EpsilonGreedyTreePerArmBandit(TreePerArmBandit):
    def __init__(self, n_actions: int, epsilon=0.9, *args, **kwargs):
        super().__init__(n_actions, *args, **kwargs)
        self.epsilon = epsilon
        self.__name__ = f"{self.__class__.__name__}_{epsilon}"

    def policy(self, rewards: list[float]) -> int:
        e = np.random.uniform(0, 1)
        if e < self.epsilon:
            return np.argmax(rewards)
        else:
            return np.random.choice(range(len(rewards)))


class SoftmaxTreePerArmBandit(TreePerArmBandit):
    def __init__(self, n_actions: int, temperature=0.2, *args, **kwargs):
        super().__init__(n_actions, *args, **kwargs)
        self.temperature = temperature
        self.__name__ = f"{self.__class__.__name__}_{temperature}"

    def policy(self, rewards: list[float]):
        probabilities = softmax(rewards, temperature=self.temperature)
        return np.random.choice(range(len(rewards)), p=probabilities)


def softmax(lst, temperature=1.0):
    lst = np.array(lst) / temperature
    exps = np.exp(lst)
    return exps / np.sum(exps)
