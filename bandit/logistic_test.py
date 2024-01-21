from .logistic import LogisticBandit, LogisticPerArmBandit


def test_logistic_bandit_pull():
    model = LogisticBandit(10)
    state = {"user": "john", "page": "home"}
    rewards = model.pull(state)
    assert rewards.shape == (10,)


def test_logistic_bandit_update():
    model = LogisticBandit(10)
    state = {"user": "john", "page": "home"}
    action = 1  # e.g. user clicked on the second button
    model.update(state, action, 1)
    model.update(state, action, -1)
    assert len(model.rewards) == 2
    rewards = model.pull(state)
    assert rewards.shape == (10,)


def test_logistic_per_arm_bandit():
    model = LogisticPerArmBandit(10)
    state = {"user": "john", "page": "home"}
    rewards = model.pull(state)
    assert rewards.shape == (10,)


def test_logistic_per_arm_bandit_update():
    model = LogisticPerArmBandit(10)
    state = {"user": "john", "page": "home"}
    action = 1  # e.g. user clicked on the second button
    model.update(state, action, 1)
    model.update(state, action, -1)
    assert len(model.rewards[action]) == 2
    rewards = model.pull(state)
    assert rewards.shape == (10,)
