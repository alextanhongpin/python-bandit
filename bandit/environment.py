import numpy as np
from itertools import product

USER_LIKED_ARTICLE = 1.0
USER_DISLIKED_ARTICLE = -1.0

users = ["Tom", "Anna"]
times_of_day = ["morning", "afternoon"]
actions = ["politics", "sports", "music", "food", "finance", "health", "camping"]


def get_cost(context, action):
    if action not in actions:
        raise ValueError(f"Unknown action: {action}")

    match (context["user"], context["time_of_day"], action):
        case ("Tom", "morning", "politics"):
            return USER_LIKED_ARTICLE
        case ("Tom", "afternoon", "music"):
            return USER_LIKED_ARTICLE
        case ("Anna", "morning", "sports"):
            return USER_LIKED_ARTICLE
        case ("Anna", "afternoon", "politics"):
            return USER_LIKED_ARTICLE
        case _:
            return USER_DISLIKED_ARTICLE


def get_cost_new(context, action):
    if action not in actions:
        raise ValueError(f"Unknown action: {action}")
    match (context["user"], context["time_of_day"], action):
        case ("Tom", "morning", "politics"):
            return USER_LIKED_ARTICLE
        case ("Tom", "afternoon", "sports"):
            return USER_LIKED_ARTICLE
        case ("Anna", "morning", "sports"):
            return USER_LIKED_ARTICLE
        case ("Anna", "afternoon", "sports"):
            return USER_LIKED_ARTICLE
        case _:
            return USER_DISLIKED_ARTICLE


def observe(rng=np.random.RandomState(None)):
    return {"user": rng.choice(users), "time_of_day": rng.choice(times_of_day)}


def feature_interaction(state: dict, action: int) -> np.ndarray:
    """perform feature interactions, similar to how vowpal wabbit does it.
    We create additional features which are the features in the (U)ser namespace and (A)ction
    namespaces multiplied together.
    This allows us to learn the interaction between when certain actions are good in certain times of days and for particular users.
    If we didn’t do that, the learning wouldn’t really work.
    We can see that in action below.
    """
    features = []
    features.append(f"action:{action}")
    features.extend([f"{k}:{v}" for k, v in state.items()])
    for kvs in product(state.items(), [("action", action)]):
        features.append("^".join([f"{k}:{v}" for k, v in kvs]))
    return np.array(features)


def one_hot_encode(state: dict, action: int) -> np.ndarray:
    X_users = np.zeros(len(users))
    if "user" in state:
        X_users[users.index(state["user"])] = 1

    X_tod = np.zeros(len(times_of_day))
    if "time_of_day" in state:
        X_tod[times_of_day.index(state["time_of_day"])] = 1

    X_actions = np.zeros(len(actions))
    if action >= 0:
        X_actions[action] = 1

    return np.array(list(np.concatenate([X_users, X_tod, X_actions])))
