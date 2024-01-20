from .logistic import EpsilonGreedyLogisticBandit
import numpy as np


class MockedLogisticBandit(EpsilonGreedyLogisticBandit):
    def preprocess(self, state: dict[str, str], action: str):
        return np.array([0, 1])


def test_logistic_bandit():
    # Write test code here
    model = MockedLogisticBandit()
    context = {"a": "b"}
    action = 0
    model.fit(context, action, 1)
    model.fit(context, action, -1)
    out = model.predict({"a": "b"}, ["1", "2", "3"])
    print("GOT", out)
    assert out is not None
