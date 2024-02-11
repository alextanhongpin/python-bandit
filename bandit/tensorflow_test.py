from .environment import actions
from .tensorflow import NeuralBandit, NeuralPerArmBandit


def test_neural_bandit():
    n_arms = len(actions)
    bandit = NeuralBandit(n_arms=n_arms)
    state = {"user": "Tom", "time_of_day": "morning"}
    rewards = bandit.pull(state)
    assert rewards.shape == (n_arms,)

    action = 1
    reward = 1
    bandit.update(state, action, reward)


def test_neural_per_arm_bandit():
    n_arms = len(actions)
    bandit = NeuralPerArmBandit()
    state = {"user": "Tom", "time_of_day": "morning"}
    rewards = bandit.pull(state)
    assert rewards.shape == (n_arms,)

    action = 1
    reward = 1
    bandit.update(state, action, reward)
