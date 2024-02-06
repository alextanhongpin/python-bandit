import numpy as np

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense

from sklearn.feature_extraction import FeatureHasher
from .environment import feature_interaction
from .policy import Softmax
from .base import BaseBandit


def create_model():
    model = Sequential(
        # 11 features that are one-hot-encoded.
        [
            Input(shape=(8,)),
            Dense(32, activation="elu"),
            Dense(32, activation="elu"),
            Dense(32, activation="elu"),
            Dense(1, activation="sigmoid"),
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
        preprocess=FeatureHasher(8, input_type="string", alternate_sign=True),
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
            # return self.update_all(state, action, reward)
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

    def update_all(self, state: dict, action: int, reward: int):
        """will punishing other arms help? seems like nope"""
        X = []
        y = []
        for i in range(self.n_arms):
            X.append(feature_interaction(state, i))
            y.append(reward if i == action else 0)
        X = np.array(self.preprocess.transform(X).toarray())
        y = np.array(y)
        return self.model.train_on_batch(X, y)

    def single_update(self, state: dict, action: int, reward: int):
        X = np.array(
            self.preprocess.transform([feature_interaction(state, action)]).toarray()
        )
        y = np.array([reward])
        # return self.model.fit(X, y, epochs=1, verbose=0)
        return self.model.train_on_batch(X, y)

    def pull(self, state: dict) -> np.ndarray:
        X = np.array(
            self.preprocess.transform(
                [feature_interaction(state, i) for i in range(self.n_arms)]
            ).toarray()
        )
        # Predict the action-value.
        return self.model.predict(X, verbose=0).flatten()


class NeuralPerArmBandit(BaseBandit):
    def __init__(
        self,
        *,
        models=create_models(7),
        batch=20,
        preprocess=FeatureHasher(8, input_type="string", alternate_sign=True),
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
        return np.array(
            [model.predict(X, verbose=0).flatten()[0] for model in self.models]
        )


class NeuralPolicyBandit(BaseBandit):
    def __init__(
        self,
        /,
        n_arms,
        *,
        preprocess=FeatureHasher(8, input_type="string", alternate_sign=True),
        batch=20,
    ):
        super().__init__(n_arms)
        self.model = self.create_model()
        self.n_arms = n_arms
        self.preprocess = preprocess
        self.softmax = Softmax(tau=0.2).softmax

    def update(self, state: dict, action: int, reward: int):
        X = np.array(
            self.preprocess.transform([feature_interaction(state, 0)]).toarray()
        )
        action_probs = self.model.predict(X, verbose=0).flatten()
        action_probs[action] += reward
        y = self.softmax(action_probs)
        # y = np.zeros(self.n_arms)
        # y[action] = self.loss_fn(action_probs, action, reward)
        y = np.array([y])
        return self.model.train_on_batch(X, y)

    def pull(self, state: dict) -> np.ndarray:
        X = np.array(
            self.preprocess.transform([feature_interaction(state, 0)]).toarray()
        )
        # Predict the action-value.
        return self.model.predict(X, verbose=0).flatten()

    def create_model(self):
        model = Sequential(
            [
                Input((8)),
                Dense(32, activation="relu"),
                Dense(32, activation="relu"),
                Dense(32, activation="relu"),
                Dense(self.n_arms, activation="softmax"),
            ]
        )
        # Why can't we use categorical cross entropy? Because we do not know
        # which arm is the best
        # And when the reward is -tive, there is no way to pass the meaning to
        # the one hot encoded y value.
        # Instead, we compute the MSE for each prediction value individually.
        # model.compile(loss="mse", optimizer="adam")
        model.compile(loss="categorical_crossentropy", optimizer="adam")
        return model

    @staticmethod
    def loss_fn(action_probs, action, reward):
        """based on REINFORCE policy"""
        return -(reward * np.log(action_probs[action]))
