import numpy as np

import torch

from sklearn.feature_extraction import FeatureHasher
from .environment import feature_interaction
from .base import BaseBandit

n_features = 8


def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        # torch.nn.init.xavier_uniform_(m.weight)
        # m.bias.data.fill_(0.01)
        n = m.in_features
        # y = 1.0 / np.sqrt(n)
        y = 1.0 / n
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0.01)


def create_model():
    model = torch.nn.Sequential(
        torch.nn.Linear(n_features, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 1),
    )
    return model


def create_models(n=7):
    return [create_model() for i in range(n)]


class NeuralBandit(BaseBandit):
    def __init__(
        self,
        /,
        n_arms,
        *,
        model=create_model(),
        preprocess=FeatureHasher(n_features, input_type="string", alternate_sign=True),
        batch=20,
    ):
        super().__init__(n_arms)
        self.model = model
        self.model.apply(init_weights)
        self.n_arms = n_arms
        self.preprocess = preprocess
        self.rewards = []
        self.state_actions = []
        self.batch = batch
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

    def update(self, state: dict, action: int, reward: int):
        n = len(self.rewards)
        b = self.batch
        if b <= 1:
            return self.update_single(state, action, reward)
        update = n % b == 0 and n != 0
        if not update:
            self.rewards.append(reward)
            self.state_actions.append(feature_interaction(state, action))
            return
        epochs = 5
        for i in range(epochs):
            indices = list(range(len(self.rewards)))
            np.random.shuffle(indices)
            X = torch.Tensor(self.preprocess.transform(self.state_actions).toarray())[
                indices
            ]
            y_pred = self.model(X)
            y = torch.Tensor(self.rewards)[indices]
            loss = self.loss_fn(y_pred, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.rewards = []
        self.state_actions = []

    def update_single(self, state: dict, action: int, reward: int):
        X = torch.Tensor(
            self.preprocess.transform([feature_interaction(state, action)]).toarray()
        )
        y_pred = self.model(X)
        y = torch.Tensor([[reward + np.random.random() / 100.0]])
        loss = self.loss_fn(y_pred, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def pull(self, state: dict) -> np.ndarray:
        X = torch.Tensor(
            self.preprocess.transform(
                [feature_interaction(state, i) for i in range(self.n_arms)]
            ).toarray()
        )

        # Predict the action-value.
        return np.array(self.model(X).data.detach()).flatten()


class NeuralPerArmBandit(BaseBandit):
    def __init__(
        self,
        *,
        models=create_models(),
        preprocess=FeatureHasher(8, input_type="string", alternate_sign=True),
        batch=20,
    ):
        n_arms = len(models)
        super().__init__(n_arms)
        self.models = models
        for model in self.models:
            model.apply(init_weights)
        self.preprocess = preprocess
        self.rewards = []
        self.state_actions = []
        self.batch = batch
        self.loss_fn = torch.nn.MSELoss()
        self.optimizers = [
            torch.optim.Adam(model.parameters(), lr=0.01) for model in self.models
        ]

    def update(self, state: dict, action: int, reward: int):
        X = torch.Tensor(
            self.preprocess.transform([feature_interaction(state, 0)]).toarray()
        )
        y_pred = self.models[action](X)
        y = torch.Tensor([[reward]])
        loss = self.loss_fn(y_pred, y)
        self.optimizers[action].zero_grad()
        loss.backward()
        self.optimizers[action].step()

    def pull(self, state: dict) -> np.ndarray:
        X = torch.Tensor(
            self.preprocess.transform([feature_interaction(state, 0)]).toarray()
        )

        # Predict the action-value.
        return np.array(
            [np.array(model(X).data.detach()).flatten() for model in self.models]
        ).flatten()
