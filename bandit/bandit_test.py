from .bandit import Bandit, PerArmBandit
from sklearn.linear_model import LogisticRegression


def test_logistic_bandit_pull():
    bandit = Bandit(LogisticRegression(random_state=42), n_arms=10)
    state = {"user": "john", "page": "home"}
    rewards = bandit.pull(state)
    assert rewards.shape == (10,)


def test_logistic_bandit_update():
    bandit = Bandit(LogisticRegression(random_state=42), n_arms=10)
    state = {"user": "john", "page": "home"}
    action = 1  # e.g. user clicked on the second button
    bandit.update(state, action, 1)
    bandit.update(state, action, -1)
    assert len(bandit.rewards) == 2
    rewards = bandit.pull(state)
    assert rewards.shape == (10,)


def test_logistic_per_arm_bandit():
    bandit = PerArmBandit([LogisticRegression(random_state=42) for _ in range(10)])
    state = {"user": "john", "page": "home"}
    rewards = bandit.pull(state)
    assert rewards.shape == (10,)


def test_logistic_per_arm_bandit_update():
    bandit = PerArmBandit([LogisticRegression(random_state=42) for _ in range(10)])
    state = {"user": "john", "page": "home"}
    action = 1  # e.g. user clicked on the second button
    bandit.update(state, action, 1)
    bandit.update(state, action, -1)
    assert len(bandit.rewards[action]) == 2
    rewards = bandit.pull(state)
    assert rewards.shape == (10,)
