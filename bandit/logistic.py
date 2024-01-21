# Adaptation of this implementation here.
# https://renan-cunha.github.io/categories/contextual-bandits/
import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import LogisticRegression
from .base import BaseBandit, extract_features


class LogisticBandit(BaseBandit):
    def __init__(
        self,
        n_arms,
        /,
        *args,
        seed=None,
        preprocess=FeatureHasher(100, input_type="string"),
        **kwargs,
    ):
        super().__init__(n_arms)
        self.rng = np.random.RandomState(seed)

        kwargs.update(random_state=seed)
        self.model = LogisticRegression(*args, **kwargs)

        self.rewards = []
        self.state_actions = []
        self.preprocess = preprocess

    def update(self, state: dict, action: int, reward: int):
        self.state_actions.append(extract_features(state, action))
        self.rewards.append(reward)

        # Need at least 2 class sample to fit the model.
        if len(np.unique(self.rewards)) < 2 and len(self.rewards) % 20 != 0:
            return
        X = self.preprocess.transform(self.state_actions)
        y = self.rewards
        self.model.fit(X, y)

    def pull(self, state: dict) -> np.ndarray:
        try:
            X = self.preprocess.transform(
                [extract_features(state, i) for i in range(self.n_arms)]
            )
            return self.model.predict_proba(X)[
                :, -1
            ]  # The last class should be the positive class.
        except NotFittedError:
            # This is the recommended way to generate random distribution.
            # If you are wondering why we did not use random.uniform():
            # https://stackoverflow.com/questions/47231852/np-random-rand-vs-np-random-random
            return self.rng.random(self.n_arms)


class LogisticPerArmBandit(BaseBandit):
    def __init__(
        self, n_arms, /, *args, seed=None, preprocess=FeatureHasher(100), **kwargs
    ):
        super().__init__(n_arms)
        self.rng = np.random.RandomState(seed)

        kwargs.update(random_state=seed)
        self.models = [LogisticRegression(*args, **kwargs) for _ in range(n_arms)]

        self.rewards = {i: [] for i in range(n_arms)}
        self.state_actions = {i: [] for i in range(n_arms)}
        self.preprocess = preprocess

    def update(self, state: dict, action: int, reward: int):
        self.state_actions[action].append(state)
        self.rewards[action].append(reward)
        # Only train if we have at least 2 targets.
        # We should also limit the training per timestep, e.g. every
        # N-iterations etc.
        X = self.preprocess.transform(self.state_actions[action])
        y = self.rewards[action]
        self.models[action].fit(X, y)

    def pull(self, state: dict) -> np.ndarray:
        rewards = np.zeros(self.n_arms)
        for action in range(self.n_arms):
            try:
                X = self.preprocess.transform([state])
                y = self.models[action].predict(X.reshape(1, -1))
                rewards[action] = y[0]
            except NotFittedError:
                # Add data with equal probability.
                self.state_actions[action].append(state)
                self.state_actions[action].append(state)
                self.rewards[action].append(0)
                self.rewards[action].append(1)

                rewards[action] = self.rng.random()
        return rewards
