import pytest
from .policy import EGreedy, Softmax, get_policy


def test_softmax():
    policy = Softmax(tau=0.1)
    vals = [-1, -1, -1, -1]
    probs = policy.softmax(vals)
    assert set(probs) == {0.25}


def test_egreedy():
    policy = EGreedy(epsilon=0.1)
    vals = [1, 2, 3]
    assert policy.policy(vals) in range(len(vals))


def test_get_policy():
    assert get_policy(EGreedy(epsilon=0.1)).__class__.__name__ == "EGreedy"
    assert get_policy("egreedy").__class__.__name__ == "EGreedy"
    assert get_policy(Softmax(tau=0.2)).__class__.__name__ == "Softmax"
    assert get_policy("softmax").__class__.__name__ == "Softmax"
    with pytest.raises(ValueError) as exc_info:
        get_policy("unknown")
    assert "Unknown policy: unknown" in str(exc_info)
