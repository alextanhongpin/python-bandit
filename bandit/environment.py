from sklearn.feature_extraction import FeatureHasher
import random
from tqdm import tqdm

USER_LIKED_ARTICLE = 1.0
USER_DISLIKED_ARTICLE = -1.0

users = ["Tom", "Anna"]
times_of_day = ["morning", "afternoon"]
actions = ["politics", "sports", "music", "food", "finance", "health", "camping"]


def get_cost(context, action):
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


def random_context(n):
    for i in tqdm(range(n)):
        yield (
            i,
            {"user": random.choice(users), "time_of_day": random.choice(times_of_day)},
        )


def preprocess(
    context, action, hasher=FeatureHasher(n_features=100, input_type="string")
):
    """perform feature interactions, similar to how vowpal wabbit does it.
    We create additional features which are the features in the (U)ser namespace and (A)ction
    namespaces multiplied together.
    This allows us to learn the interaction between when certain actions are good in certain times of days and for particular users.
    If we didn’t do that, the learning wouldn’t really work.
    We can see that in action below.
    """
    user = context["user"]
    time_of_day = context["time_of_day"]
    features = [
        f"user:{user}",
        f"action:{action}^user:{user}",
        f"time_of_day:{time_of_day}^user:{user}",
    ]
    return hasher.transform([features]).toarray()[0]