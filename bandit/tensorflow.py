import numpy as np

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense

# from tensorflow.keras.optimizers import SGD
from sklearn.feature_extraction import FeatureHasher
from .environment import feature_interaction
from .base import BaseBandit


def create_model():
    model = Sequential(
        # 11 features that are one-hot-encoded.
        [
            Input(shape=(10,)),
            Dense(32, activation="relu"),
            # Dropout(0.1),
            Dense(32, activation="relu"),
            # Dropout(0.1),
            Dense(1),
        ]
    )
    # optimizer = SGD(learning_rate=0.001)
    model.compile(
        loss="mse",
        optimizer="adam",
        metrics=[
            "mse",
        ],
    )
    return model


def create_models(n):
    return [create_model() for i in range(n)]


class NeuralBandit(BaseBandit):
    def __init__(
        self,
        /,
        n_arms,
        *,
        model=create_model(),
        preprocess=FeatureHasher(10, input_type="string", alternate_sign=True),
        batch=20,
    ):
        super().__init__(n_arms)
        self.model = model
        self.n_arms = n_arms
        self.preprocess = preprocess
        self.rewards = []
        self.state_actions = []
        self.batch = batch

    def update(self, state: dict, action: int, reward: int):
        if self.batch == 1:
            return self.single_update(state, action, reward)

        self.rewards.append(reward)
        self.state_actions.append(feature_interaction(state, action))

        if len(self.rewards) % self.batch == 0 and len(self.rewards) != 0:
            X = np.array(self.preprocess.transform(self.state_actions).toarray())
            # X = np.array(self.state_actions)
            y = np.array(self.rewards)

            self.state_actions = []
            self.rewards = []
            # Keras 3 has fit_on_batch and predict_on_batch.
            # However, at the time of writing, tensorflow version is only 2.15,
            # while keras3.0 is only available for tensorflow version 2.16+.
            return self.model.fit(X, y)

    def single_update(self, state: dict, action: int, reward: int):
        X = np.array(
            self.preprocess.transform([feature_interaction(state, action)]).toarray()
        )
        y = np.array([reward])
        return self.model.train_on_batch(X, y)

    def pull(self, state: dict) -> np.ndarray:
        X = self.preprocess.transform(
            [feature_interaction(state, i) for i in range(self.n_arms)]
        )
        # X = np.array([feature_interaction(state, i) for i in range(self.n_arms)])
        return self.model.predict(X, verbose=0)


class NeuralPerArmBandit(BaseBandit):
    def __init__(
        self,
        *,
        models=create_models(7),
        batch=20,
        preprocess=FeatureHasher(10, input_type="string", alternate_sign=True),
    ):
        n_arms = len(models)
        super().__init__(n_arms)
        self.models = models
        self.n_arms = n_arms
        self.preprocess = preprocess
        self.rewards = {i: [] for i in range(n_arms)}
        self.state_actions = {i: [] for i in range(n_arms)}
        self.batch = batch

    def update(self, state: dict, action: int, reward: int):
        if self.batch == 1:
            return self.single_update(state, action, reward)

        self.rewards[action].append(reward)
        self.state_actions[action].append(feature_interaction(state, -1))

        # X = np.array(self.state_actions[action])
        X = np.array(self.preprocess.transform(self.state_actions[action]).toarray())
        y = np.array(self.rewards[action])

        if len(y) % self.batch == 0 and len(y) != 0:
            self.state_actions[action] = []
            self.rewards[action] = []
            self.models[action].fit(X, y, epochs=1, verbose=0)

    def single_update(self, state: dict, action: int, reward: int):
        X = np.array(
            self.preprocess.transform([feature_interaction(state, action)]).toarray()
        )
        y = np.array([reward])
        return self.models[action].train_on_batch(X, y)

    def pull(self, state: dict) -> np.ndarray:
        X = self.preprocess.transform([feature_interaction(state, -1)])
        # X = np.array([feature_interaction(state, -1)])
        return np.array([model.predict(X, verbose=0)[0] for model in self.models])
