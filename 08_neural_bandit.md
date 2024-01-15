# Neural Bandit

Implementing neural bandit using scikit-learn's multilayer perceptron.


```python
import random

import matplotlib.pyplot as plt
import numpy as np
```


```python
USER_LIKED_ARTICLE = 1.0
USER_DISLIKED_ARTICLE = 0.0
```


```python
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
```


```python
users = ["Tom", "Anna"]
times_of_day = ["morning", "afternoon"]
actions = ["politics", "sports", "music", "food", "finance", "health", "camping"]


def one_hot_encode(user, time_of_day, action):
    return [users.index(user) + 1, times_of_day.index(time_of_day) + 1] + [
        1 if a == action else 0 for a in actions
    ]


one_hot_encode("Tom", "morning", "politics")
```




    [1, 1, 1, 0, 0, 0, 0, 0, 0]




```python
from sklearn.feature_extraction import FeatureHasher

hasher = FeatureHasher(n_features=10, input_type="string")
print(hasher.transform([["a", "b", "c"]]).toarray())
print(hasher.transform([["c", "a", "b"]]).toarray())
```

    [[ 1. -1.  0.  0.  0.  0.  0. -1.  0.  0.]]
    [[ 1. -1.  0.  0.  0.  0.  0. -1.  0.  0.]]



```python
from itertools import combinations_with_replacement

# This is similar to how vowpal wabbit does the feature interaction.
print(list(combinations_with_replacement("abc", 2)))
print(list(combinations_with_replacement(["Tom", "politics"], 2)))
```

    [('a', 'a'), ('a', 'b'), ('a', 'c'), ('b', 'b'), ('b', 'c'), ('c', 'c')]
    [('Tom', 'Tom'), ('Tom', 'politics'), ('politics', 'politics')]



```python
# https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Feature-interactions
# https://vowpalwabbit.org/docs/vowpal_wabbit/python/latest/tutorials/python_Simulating_a_news_personalization_scenario_using_Contextual_Bandits.html#:~:text=supply%20to%20VW-,%2C%20we%20include%20%2Dq%20UA,-.%20This%20is%20telling
def feature_interaction(user, time_of_day, action):
    """perform feature interactions, similar to how vowpal wabbit does it.
    We create additional features which are the features in the (U)ser namespace and (A)ction
    namespaces multiplied together.
    This allows us to learn the interaction between when certain actions are good in certain times of days and for particular users.
    If we didn’t do that, the learning wouldn’t really work.
    We can see that in action below.
    """
    features = [
        user,
        time_of_day,
        user + ":" + action,
        time_of_day + ":" + action,
        action,
    ]
    return hasher.transform([features]).toarray()[0]
```


```python
feature_interaction("Tom", "morning", "politics")
```




    array([ 0.,  0., -1.,  1.,  0.,  0.,  2.,  0.,  1.,  0.])




```python
def sample_custom_pmf(pmf):
    total = sum(pmf)
    scale = 1 / total
    pmf = [x * scale for x in pmf]
    draw = random.random()
    sum_prob = 0.0
    for index, prob in enumerate(pmf):
        sum_prob += prob
        if sum_prob > draw:
            return index, prob
```


```python
from sklearn.neural_network import MLPRegressor

model = MLPRegressor(activation="relu", random_state=42)
model.partial_fit([one_hot_encode("Tom", "morning", "politics")], [1])
model.predict([one_hot_encode("Tom", "morning", "politics")])
```




    array([0.20727216])




```python
# Sampling best reward for the action taken.
pmf = model.predict([one_hot_encode("Tom", "morning", action) for action in actions])
action, prob = sample_custom_pmf(pmf)
pmf, action, prob
```




    (array([ 0.20727216,  0.11697445,  0.33291172,  0.24473987,  0.35555384,
             0.13795523, -0.00638099]),
     1,
     0.08421327220560895)




```python
def choose_user(users):
    return random.choice(users)


def choose_time_of_day(times_of_day):
    return random.choice(times_of_day)


def get_action(model, context, actions, preprocess):
    pmf = model.predict(
        [
            preprocess(context["user"], context["time_of_day"], action)
            for action in actions
        ]
    )
    chosen_action_index, prob = sample_custom_pmf(pmf)
    return actions[chosen_action_index], prob
```


```python
def run_simulation(
    model,
    num_iterations,
    users,
    times_of_day,
    actions,
    cost_function,
    do_learn=True,
    preprocess=one_hot_encode,
):
    cost_sum = 0.0
    ctr = []

    for i in range(1, num_iterations + 1):
        # 1. In each simulation choose a user
        user = choose_user(users)
        # 2. Choose time of day for a given user
        time_of_day = choose_time_of_day(times_of_day)

        # 3. Pass context to vw to get an action
        context = {"user": user, "time_of_day": time_of_day}
        action, prob = get_action(model, context, actions, preprocess)

        # 4. Get cost of the action we chose
        cost = cost_function(context, action)
        cost_sum += cost

        if do_learn:
            # 5. Learn
            model.partial_fit([preprocess(user, time_of_day, action)], [cost])

        # We negate this so that on the plot instead of minimizing cost, we are maximizing reward
        ctr.append(1 * cost_sum / i)

    return ctr
```


```python
def plot_ctr(num_iterations, ctr):
    plt.plot(range(1, num_iterations + 1), ctr)
    plt.xlabel("num_iterations", fontsize=14)
    plt.ylabel("ctr", fontsize=14)
    plt.ylim([0, 1])
```


```python
num_iterations = 5000
# Need to fit at least once.
model = MLPRegressor(activation="relu", random_state=42)
model.partial_fit([one_hot_encode("Tom", "morning", "politics")], [1])
ctr = run_simulation(model, num_iterations, users, times_of_day, actions, get_cost)
old_ctr = ctr
plot_ctr(num_iterations, ctr)
```


    
![png](08_neural_bandit_files/08_neural_bandit_15_0.png)
    



```python
num_iterations = 5000
# Need to fit at least once.
model = MLPRegressor(activation="relu", random_state=42)
model.partial_fit([feature_interaction("Tom", "morning", "politics")], [1])
ctr = run_simulation(
    model,
    num_iterations,
    users,
    times_of_day,
    actions,
    get_cost,
    preprocess=feature_interaction,
)

plt.plot(range(1, num_iterations + 1), old_ctr)
plot_ctr(num_iterations, ctr)
plt.legend(["one_hot_encode", "feature_interaction"])
```




    <matplotlib.legend.Legend at 0x1197b49a0>




    
![png](08_neural_bandit_files/08_neural_bandit_16_1.png)
    



```python
context = {"user": "Anna", "time_of_day": "morning"}
action, prob = get_action(model, context, actions, feature_interaction)
action, prob
```




    ('sports', 1.1332483854814464)




```python
context = {"user": "Anna", "time_of_day": "afternoon"}
action, prob = get_action(model, context, actions, feature_interaction)
action, prob
```




    ('politics', 0.9507944715662973)




```python
def get_cost_new1(context, action):
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
```


```python
def run_simulation_multiple_cost_functions(
    model,
    num_iterations,
    users,
    times_of_day,
    actions,
    cost_functions,
    do_learn=True,
    preprocess=feature_interaction,
):
    cost_sum = 0.0
    ctr = []

    start_counter = 1
    end_counter = start_counter + num_iterations
    for cost_function in cost_functions:
        for i in range(start_counter, end_counter):
            # 1. in each simulation choose a user
            user = choose_user(users)
            # 2. choose time of day for a given user
            time_of_day = choose_time_of_day(times_of_day)

            # Construct context based on chosen user and time of day
            context = {"user": user, "time_of_day": time_of_day}

            # 3. Use the get_action function we defined earlier
            action, prob = get_action(model, context, actions, preprocess)

            # 4. Get cost of the action we chose
            cost = cost_function(context, action)
            cost_sum += cost

            if do_learn:
                # 5. Inform VW of what happened so we can learn from it
                model.partial_fit([preprocess(user, time_of_day, action)], [cost])

            ctr.append(1 * cost_sum / i)
        start_counter = end_counter
        end_counter = start_counter + num_iterations

    return ctr
```


```python
model = MLPRegressor(activation="relu", random_state=42)
# Need to fit at least one data before using.\n",
model.partial_fit([feature_interaction("Tom", "morning", "politics")], [1])
ctr = run_simulation_multiple_cost_functions(
    model,
    num_iterations_per_cost_func,
    users,
    times_of_day,
    actions,
    cost_functions,
    preprocess=feature_interaction,
)
plot_ctr(total_iterations, ctr)
```


    
![png](08_neural_bandit_files/08_neural_bandit_21_0.png)
    

