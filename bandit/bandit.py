# Adaptation of this implementation here.
# https://renan-cunha.github.io/categories/contextual-bandits/
import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction import FeatureHasher
from .base import BaseBandit
from .environment import feature_interaction


class Bandit(BaseBandit):
    def __init__(
        self,
        model,
        /,
        n_arms,
        *,
        seed=None,
        preprocess=FeatureHasher(100, input_type="string"),
    ):
        super().__init__(n_arms)
        self.rng = np.random.RandomState(seed)

        self.model = model

        self.rewards = []
        self.state_actions = []
        self.preprocess = preprocess

    def update(self, state: dict, action: int, reward: int):
        if hasattr(self.model, "partial_fit"):
            self.model.partial_fit(
                self.preprocess.transform([feature_interaction(state, action)]),
                [reward],
            )
            return

        self.state_actions.append(feature_interaction(state, action))
        self.rewards.append(reward)
        # Need at least 2 class sample to fit the model.
        if len(np.unique(self.rewards)) < 2 or len(self.rewards) % 20 != 0:
            return
        X = self.preprocess.transform(self.state_actions)
        y = self.rewards
        self.model.fit(X, y)

    def pull(self, state: dict) -> np.ndarray:
        try:
            X = self.preprocess.transform(
                [feature_interaction(state, i) for i in range(self.n_arms)]
            )
            if hasattr(self.model, "predict_proba"):
                # The last class should be the positive class.
                return self.model.predict_proba(X)[:, -1]
            return self.model.predict(X)
        except NotFittedError:
            # This is the recommended way to generate random distribution.
            # If you are wondering why we did not use random.uniform():
            # https://stackoverflow.com/questions/47231852/np-random-rand-vs-np-random-random
            return self.rng.random(self.n_arms)


class PerArmBandit(BaseBandit):
    def __init__(
        self,
        models,
        /,
        seed=None,
        preprocess=FeatureHasher(100),
    ):
        n_arms = len(models)
        super().__init__(n_arms)
        self.rng = np.random.RandomState(seed)

        self.models = models
        self.rewards = {i: [] for i in range(n_arms)}
        self.state_actions = {i: [] for i in range(n_arms)}
        self.preprocess = preprocess

    def update(self, state: dict, action: int, reward: int):
        if hasattr(self.models[action], "partial_fit"):
            self.models[action].partial_fit(
                self.preprocess.transform([state]), [reward]
            )
            return

        self.state_actions[action].append(state)
        self.rewards[action].append(reward)

        # Only train if we have at least 2 targets.
        # We should also limit the training per timestep, e.g. every
        # N-iterations etc.
        if (
            len(np.unique(self.rewards[action])) < 2
            or len(self.rewards[action]) % 20 != 0
        ):
            return
        X = self.preprocess.transform(self.state_actions[action])
        y = self.rewards[action]
        self.models[action].fit(X, y)

    def pull(self, state: dict) -> np.ndarray:
        rewards = np.zeros(self.n_arms)
        for action in range(self.n_arms):
            try:
                model = self.models[action]
                X = self.preprocess.transform([state]).reshape(1, -1)

                if hasattr(model, "predict_proba"):
                    # The last class should be the positive class.
                    y = model.predict_proba(X)[0][-1]
                else:
                    y = model.predict(X)[0]
                rewards[action] = y
            except NotFittedError:
                # Add data with equal probability.
                self.state_actions[action].append(state)
                self.state_actions[action].append(state)
                self.rewards[action].append(0)
                self.rewards[action].append(1)

                rewards[action] = self.rng.random()
        return rewards
