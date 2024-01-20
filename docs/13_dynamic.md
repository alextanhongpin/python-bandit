```python
import json
import os
from timeit import default_timer as timer

import pandas as pd
import seaborn as sns

import bandit.environment as env
from bandit.logistic import (
    EpsilonGreedyLogisticBandit,
    EpsilonGreedyLogisticPerArmBandit,
    SoftmaxLogisticBandit,
    SoftmaxLogisticPerArmBandit,
)
from bandit.neural_bandit import (
    EpsilonGreedyNeuralBandit,
    EpsilonGreedyNeuralPerArmBandit,
    SoftmaxNeuralBandit,
    SoftmaxNeuralPerArmBandit,
)
from bandit.tree import (
    EpsilonGreedyTreeBandit,
    EpsilonGreedyTreePerArmBandit,
    EpsilonGreedyTreeSamplingBandit,
    SoftmaxTreeBandit,
    SoftmaxTreePerArmBandit,
    SoftmaxTreeSamplingBandit,
)

sns.set_theme()
```


```python
import json

json_file_name = "contextual_bandit_dynamic.json"
with open(json_file_name, "a+") as f:
    # It goes to the last line
    f.seek(0)
    try:
        data = json.load(f)
    except:
        data = {}
len(data)
```




    28




```python
def save_json():
    with open(json_file_name, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    return data
```


```python
import importlib

import bandit

importlib.reload(bandit.neural_bandit)
importlib.reload(bandit.logistic)
```




    <module 'bandit.logistic' from '/Users/alextanhongpin/Documents/python/python-bandit/bandit/logistic.py'>




```python
class ContextualEpsilonGreedyNeuralBandit(EpsilonGreedyNeuralBandit):
    def __init__(self, actions, epsilon=1.0, *args, **kwargs):
        super().__init__(epsilon, random_state=42, *args, **kwargs)
        self.actions = actions

    def preprocess(self, state: dict[str, str], action):
        return env.preprocess(state, action)

    def predict(self, state: dict[str, str]):
        return super().predict(state, env.actions)


class ContextualEpsilonGreedyNeuralPerArmBandit(EpsilonGreedyNeuralPerArmBandit):
    def __init__(self, actions, epsilon=1.0, *args, **kwargs):
        super().__init__(len(actions), epsilon, random_state=42, *args, **kwargs)
        self.actions = actions

    def fit(self, state: dict[str, str], action: str, reward: float):
        super().fit(state, env.actions.index(action), reward)

    def preprocess(self, state: dict[str, str]):
        action = ""
        return env.preprocess(state, action)

    def predict(self, state: dict[str, str]):
        return super().predict(state, env.actions)


class ContextualEpsilonGreedyLogisticBandit(EpsilonGreedyLogisticBandit):
    def __init__(self, actions, epsilon=1.0, *args, **kwargs):
        super().__init__(epsilon, random_state=42, *args, **kwargs)
        self.actions = actions

    def preprocess(self, state: dict[str, str], action):
        return env.preprocess(state, action)

    def predict(self, state: dict[str, str]):
        return super().predict(state, env.actions)


class ContextualEpsilonGreedyLogisticPerArmBandit(EpsilonGreedyLogisticPerArmBandit):
    def __init__(self, actions, epsilon=1.0, *args, **kwargs):
        super().__init__(len(actions), epsilon, random_state=42, *args, **kwargs)
        self.actions = actions

    def fit(self, state: dict[str, str], action: str, reward: float):
        super().fit(state, env.actions.index(action), reward)

    def preprocess(self, state: dict[str, str]):
        action = ""
        return env.one_hot_encode(state, action)

    def predict(self, state: dict[str, str]):
        return super().predict(state, env.actions)


class ContextualEpsilonGreedyTreeBandit(EpsilonGreedyTreeBandit):
    def __init__(self, actions, epsilon=1.0, *args, **kwargs):
        super().__init__(epsilon, random_state=42, *args, **kwargs)
        self.actions = actions

    def preprocess(self, state: dict[str, str], action):
        return env.preprocess(state, action)

    def predict(self, state: dict[str, str]):
        return super().predict(state, env.actions)


class ContextualEpsilonGreedyTreeSamplingBandit(EpsilonGreedyTreeSamplingBandit):
    def __init__(self, actions, epsilon=1.0, *args, **kwargs):
        super().__init__(epsilon, random_state=42, *args, **kwargs)
        self.actions = actions

    def preprocess(self, state: dict[str, str], action):
        return env.preprocess(state, action)

    def predict(self, state: dict[str, str]):
        return super().predict(state, env.actions)


class ContextualEpsilonGreedyTreePerArmBandit(EpsilonGreedyTreePerArmBandit):
    def __init__(self, actions, epsilon=1.0, *args, **kwargs):
        super().__init__(len(actions), epsilon, random_state=42, *args, **kwargs)
        self.actions = actions

    def fit(self, state: dict[str, str], action: str, reward: float):
        super().fit(state, env.actions.index(action), reward)

    def preprocess(self, state: dict[str, str]):
        action = ""
        return env.one_hot_encode(state, action)

    def predict(self, state: dict[str, str]):
        return super().predict(state, env.actions)
```


```python
class ContextualSoftmaxNeuralBandit(SoftmaxNeuralBandit):
    def __init__(self, actions, temperature=1.0, *args, **kwargs):
        super().__init__(temperature, random_state=42, *args, **kwargs)
        self.actions = actions

    def preprocess(self, state, action):
        return env.preprocess(state, action)

    def predict(self, state: dict[str, str]):
        return super().predict(state, env.actions)


class ContextualSoftmaxNeuralPerArmBandit(SoftmaxNeuralPerArmBandit):
    def __init__(self, actions, temperature=1.0, *args, **kwargs):
        super().__init__(len(actions), temperature, random_state=42, *args, **kwargs)
        self.actions = actions

    def fit(self, state: dict[str, str], action: str, reward: float):
        super().fit(state, env.actions.index(action), reward)

    def preprocess(self, state: dict[str, str]):
        action = ""
        return env.preprocess(state, action)

    def predict(self, state: dict[str, str]):
        return super().predict(state, env.actions)


class ContextualSoftmaxLogisticBandit(SoftmaxLogisticBandit):
    def __init__(self, actions, temperature=1.0, *args, **kwargs):
        super().__init__(temperature, random_state=42, *args, **kwargs)
        self.actions = actions

    def preprocess(self, state, action):
        return env.preprocess(state, action)

    def predict(self, state: dict[str, str]):
        return super().predict(state, env.actions)


class ContextualSoftmaxLogisticPerArmBandit(SoftmaxLogisticPerArmBandit):
    def __init__(self, actions, temperature=1.0, *args, **kwargs):
        super().__init__(len(actions), temperature, random_state=42, *args, **kwargs)
        self.actions = actions

    def fit(self, state: dict[str, str], action: str, reward: float):
        super().fit(state, env.actions.index(action), reward)

    def preprocess(self, state: dict[str, str]):
        action = ""
        return env.one_hot_encode(state, action)

    def predict(self, state: dict[str, str]):
        return super().predict(state, env.actions)


class ContextualSoftmaxLogisticBandit(SoftmaxLogisticBandit):
    def __init__(self, actions, temperature=1.0, *args, **kwargs):
        super().__init__(temperature, random_state=42, *args, **kwargs)
        self.actions = actions

    def preprocess(self, state, action):
        return env.preprocess(state, action)

    def predict(self, state: dict[str, str]):
        return super().predict(state, env.actions)


class ContextualSoftmaxTreePerArmBandit(SoftmaxTreePerArmBandit):
    def __init__(self, actions, temperature=1.0, *args, **kwargs):
        super().__init__(len(actions), temperature, random_state=42, *args, **kwargs)
        self.actions = actions

    def fit(self, state: dict[str, str], action: str, reward: float):
        super().fit(state, env.actions.index(action), reward)

    def preprocess(self, state: dict[str, str]):
        action = ""
        return env.one_hot_encode(state, action)

    def predict(self, state: dict[str, str]):
        return super().predict(state, env.actions)


class ContextualSoftmaxTreeBandit(SoftmaxTreeBandit):
    def __init__(self, actions, temperature=1.0, *args, **kwargs):
        super().__init__(temperature, random_state=42, *args, **kwargs)
        self.actions = actions

    def preprocess(self, state, action):
        return env.preprocess(state, action)

    def predict(self, state: dict[str, str]):
        return super().predict(state, env.actions)


class ContextualSoftmaxTreeSamplingBandit(SoftmaxTreeSamplingBandit):
    def __init__(self, actions, temperature=1.0, *args, **kwargs):
        super().__init__(temperature, random_state=42, *args, **kwargs)
        self.actions = actions

    def preprocess(self, state, action):
        return env.preprocess(state, action)

    def predict(self, state: dict[str, str]):
        return super().predict(state, env.actions)
```


```python
N = 5_000
file_name = "contextual_bandit_dynamic.csv"
if not os.path.exists(file_name):
    df = pd.DataFrame(index=range(N))
    df.to_csv(file_name)
else:
    df = pd.read_csv(file_name)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ContextualEpsilonGreedyNeuralBandit_1.0</th>
      <th>ContextualEpsilonGreedyNeuralBandit_0.9</th>
      <th>ContextualEpsilonGreedyNeuralPerArmBandit_1.0</th>
      <th>ContextualEpsilonGreedyNeuralPerArmBandit_0.9</th>
      <th>ContextualEpsilonGreedyLogisticBandit_1.0</th>
      <th>ContextualEpsilonGreedyLogisticBandit_0.9</th>
      <th>ContextualEpsilonGreedyLogisticPerArmBandit_1.0</th>
      <th>ContextualEpsilonGreedyLogisticPerArmBandit_0.9</th>
      <th>ContextualEpsilonGreedyTreeBandit_1.0</th>
      <th>ContextualEpsilonGreedyTreeBandit_0.9</th>
      <th>...</th>
      <th>ContextualSoftmaxLogisticPerArmBandit_0.2</th>
      <th>ContextualSoftmaxLogisticPerArmBandit_0.5</th>
      <th>ContextualSoftmaxTreeBandit_0.2</th>
      <th>ContextualSoftmaxTreeBandit_0.5</th>
      <th>ContextualSoftmaxTreePerArmBandit_0.2</th>
      <th>ContextualSoftmaxTreePerArmBandit_0.5</th>
      <th>ContextualEpsilonGreedyTreeSamplingBandit_1.0</th>
      <th>ContextualEpsilonGreedyTreeSamplingBandit_0.9</th>
      <th>ContextualSoftmaxTreeSamplingBandit_0.2</th>
      <th>ContextualSoftmaxTreeSamplingBandit_0.5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.500000</td>
      <td>0.500000</td>
      <td>0.0</td>
      <td>0.500000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.500000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.500000</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.666667</td>
      <td>0.666667</td>
      <td>0.0</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.333333</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.333333</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.333333</td>
      <td>0.333333</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.750000</td>
      <td>0.500000</td>
      <td>0.0</td>
      <td>0.250000</td>
      <td>0.500000</td>
      <td>0.0</td>
      <td>0.25</td>
      <td>0.250000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.250000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.250000</td>
      <td>0.250000</td>
      <td>0.25</td>
      <td>0.0</td>
      <td>0.250000</td>
      <td>0.250000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.800000</td>
      <td>0.600000</td>
      <td>0.0</td>
      <td>0.200000</td>
      <td>0.400000</td>
      <td>0.0</td>
      <td>0.20</td>
      <td>0.200000</td>
      <td>0.0</td>
      <td>0.2</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.400000</td>
      <td>0.2</td>
      <td>0.0</td>
      <td>0.200000</td>
      <td>0.400000</td>
      <td>0.20</td>
      <td>0.0</td>
      <td>0.200000</td>
      <td>0.200000</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 28 columns</p>
</div>




```python
def run_simulation(model):
    total_reward = 0
    avg_rewards = []

    for i, ctx in env.random_context(N, random_state=42):
        # 1. Predict the action.
        action = model.predict(ctx)

        # 2. Act and get the reward.
        cost_fn = env.get_cost
        if i > N // 2:
            cost_fn = env.get_cost_new1
        reward = cost_fn(ctx, action)

        # 3. Update the model.
        model.fit(ctx, action, reward)

        # 4. Save the reward.
        total_reward += max(0, reward)
        avg_rewards.append(total_reward / (i + 1))
    return avg_rewards, total_reward
```


```python
models = [
    # EpsilonGreedy.
    ContextualEpsilonGreedyNeuralBandit(env.actions, epsilon=1.0),
    ContextualEpsilonGreedyNeuralBandit(env.actions, epsilon=0.9),
    ContextualEpsilonGreedyNeuralPerArmBandit(env.actions, epsilon=1.0),
    ContextualEpsilonGreedyNeuralPerArmBandit(env.actions, epsilon=0.9),
    ContextualEpsilonGreedyLogisticBandit(env.actions, epsilon=1.0),
    ContextualEpsilonGreedyLogisticBandit(env.actions, epsilon=0.9),
    ContextualEpsilonGreedyLogisticPerArmBandit(env.actions, epsilon=1.0),
    ContextualEpsilonGreedyLogisticPerArmBandit(env.actions, epsilon=0.9),
    ContextualEpsilonGreedyTreeBandit(env.actions, epsilon=1.0),
    ContextualEpsilonGreedyTreeBandit(env.actions, epsilon=0.9),
    ContextualEpsilonGreedyTreePerArmBandit(env.actions, epsilon=1.0),
    ContextualEpsilonGreedyTreePerArmBandit(env.actions, epsilon=0.9),
    ContextualEpsilonGreedyTreeSamplingBandit(env.actions, epsilon=1.0),
    ContextualEpsilonGreedyTreeSamplingBandit(env.actions, epsilon=0.9),
    # Softmax.
    ContextualSoftmaxNeuralBandit(env.actions, temperature=0.2),
    ContextualSoftmaxNeuralBandit(env.actions, temperature=0.5),
    ContextualSoftmaxNeuralPerArmBandit(env.actions, temperature=0.2),
    ContextualSoftmaxNeuralPerArmBandit(env.actions, temperature=0.5),
    ContextualSoftmaxLogisticBandit(env.actions, temperature=0.2),
    ContextualSoftmaxLogisticBandit(env.actions, temperature=0.5),
    ContextualSoftmaxLogisticPerArmBandit(env.actions, temperature=0.2),
    ContextualSoftmaxLogisticPerArmBandit(env.actions, temperature=0.5),
    ContextualSoftmaxTreeBandit(env.actions, temperature=0.2),
    ContextualSoftmaxTreeBandit(env.actions, temperature=0.5),
    ContextualSoftmaxTreePerArmBandit(env.actions, temperature=0.2),
    ContextualSoftmaxTreePerArmBandit(env.actions, temperature=0.5),
    ContextualSoftmaxTreeSamplingBandit(env.actions, temperature=0.2),
    ContextualSoftmaxTreeSamplingBandit(env.actions, temperature=0.5),
]
# TODO: Benchmark time.
for model in models:
    if (name := model.__name__) in df:
        print("skipping", name)
        # if "Sampling" not in name:
        #     continue
        continue
    start = timer()
    avg_rewards, total_reward = run_simulation(model)
    elapsed = timer() - start
    if name not in data:
        data[name] = {}
    data[name]["elapsed"] = elapsed
    data[name]["total_reward"] = total_reward
    df[name] = avg_rewards
    df.to_csv(file_name, index=False)
    save_json()
```

    skipping ContextualEpsilonGreedyNeuralBandit_1.0
    skipping ContextualEpsilonGreedyNeuralBandit_0.9
    skipping ContextualEpsilonGreedyNeuralPerArmBandit_1.0
    skipping ContextualEpsilonGreedyNeuralPerArmBandit_0.9
    skipping ContextualEpsilonGreedyLogisticBandit_1.0
    skipping ContextualEpsilonGreedyLogisticBandit_0.9
    skipping ContextualEpsilonGreedyLogisticPerArmBandit_1.0
    skipping ContextualEpsilonGreedyLogisticPerArmBandit_0.9
    skipping ContextualEpsilonGreedyTreeBandit_1.0
    skipping ContextualEpsilonGreedyTreeBandit_0.9
    skipping ContextualEpsilonGreedyTreePerArmBandit_1.0
    skipping ContextualEpsilonGreedyTreePerArmBandit_0.9
    skipping ContextualEpsilonGreedyTreeSamplingBandit_1.0
    skipping ContextualEpsilonGreedyTreeSamplingBandit_0.9
    skipping ContextualSoftmaxNeuralBandit_0.2
    skipping ContextualSoftmaxNeuralBandit_0.5
    skipping ContextualSoftmaxNeuralPerArmBandit_0.2
    skipping ContextualSoftmaxNeuralPerArmBandit_0.5
    skipping ContextualSoftmaxLogisticBandit_0.2
    skipping ContextualSoftmaxLogisticBandit_0.5
    skipping ContextualSoftmaxLogisticPerArmBandit_0.2
    skipping ContextualSoftmaxLogisticPerArmBandit_0.5
    skipping ContextualSoftmaxTreeBandit_0.2
    skipping ContextualSoftmaxTreeBandit_0.5
    skipping ContextualSoftmaxTreePerArmBandit_0.2
    skipping ContextualSoftmaxTreePerArmBandit_0.5
    skipping ContextualSoftmaxTreeSamplingBandit_0.2
    skipping ContextualSoftmaxTreeSamplingBandit_0.5



```python
# Override
# You will get unnamed columns if saved with index=True
# df[[col for col in df.columns if 'Unnamed' not in col]]
# df.to_csv(file_name, index=False)
```


```python
def plot(*patterns):
    cols = df.columns
    for pat in patterns:
        cols = list(filter(lambda col: pat in col, cols))

    style = []
    for col in cols:
        if "NeuralBandit" in col:
            style.append("-")
        elif "NeuralBanditPerArm" in col:
            style.append(":")
        elif "Tree" in col:
            style.append("--")
        elif "Logistic" in col:
            style.append("-.")
        else:
            style.append("-")
    df[cols].plot(
        figsize=(12, 8),
        style=style,
        xlabel="Trials",
        ylabel="Avg. Rewards",
        title="Contextual Bandit",
    )
```


```python
plot()
```


    
![png](13_dynamic_files/13_dynamic_11_0.png)
    



```python
plot("Softmax")
```


    
![png](13_dynamic_files/13_dynamic_12_0.png)
    



```python
plot("SoftmaxNeuralBandit")
```


    
![png](13_dynamic_files/13_dynamic_13_0.png)
    



```python
plot("SoftmaxNeuralPerArmBandit")
```


    
![png](13_dynamic_files/13_dynamic_14_0.png)
    



```python
plot("Softmax", "0.2")
```


    
![png](13_dynamic_files/13_dynamic_15_0.png)
    



```python
plot("Softmax", "0.5")
```


    
![png](13_dynamic_files/13_dynamic_16_0.png)
    



```python
plot("Softmax", "Tree", "0.2")
```


    
![png](13_dynamic_files/13_dynamic_17_0.png)
    



```python
plot("Greedy")
```


    
![png](13_dynamic_files/13_dynamic_18_0.png)
    



```python
plot("Greedy", "Tree", "0.9")
```


    
![png](13_dynamic_files/13_dynamic_19_0.png)
    



```python
plot("GreedyNeuralBandit")
```


    
![png](13_dynamic_files/13_dynamic_20_0.png)
    



```python
plot("GreedyNeuralPerArmBandit")
```


    
![png](13_dynamic_files/13_dynamic_21_0.png)
    



```python
plot("Greedy", "0.9")
```


    
![png](13_dynamic_files/13_dynamic_22_0.png)
    



```python
plot("Greedy", "1.0")
```


    
![png](13_dynamic_files/13_dynamic_23_0.png)
    



```python
plot("GreedyNeural", "1.0")
```


    
![png](13_dynamic_files/13_dynamic_24_0.png)
    



```python
plot("NeuralBandit")
```


    
![png](13_dynamic_files/13_dynamic_25_0.png)
    



```python
plot("NeuralPerArmBandit")
```


    
![png](13_dynamic_files/13_dynamic_26_0.png)
    



```python
plot("Logistic")
```


    
![png](13_dynamic_files/13_dynamic_27_0.png)
    



```python
plot("Logistic", "Epsilon", "PerArm")
```


    
![png](13_dynamic_files/13_dynamic_28_0.png)
    



```python
plot("Logistic", "Epsilon")
```


    
![png](13_dynamic_files/13_dynamic_29_0.png)
    



```python
plot("Logistic", "Softmax")
```


    
![png](13_dynamic_files/13_dynamic_30_0.png)
    



```python
plot("Tree")
```


    
![png](13_dynamic_files/13_dynamic_31_0.png)
    



```python
model = models[-1]
# model = ContextualEpsilonGreedyTreeBandit(env.actions, epsilon=0.9)

# _, total_reward = run_simulation(model)
# total_reward
```


```python
ctx = {"user": "Tom", "time_of_day": "morning"}
# 1. Predict the action.
action = model.predict(ctx)

# 2. Act and get the reward.
reward = env.get_cost(ctx, action)
print("Context:", ctx)
print("Action:", action)
print("Reward:", reward)
```

    Context: {'user': 'Tom', 'time_of_day': 'morning'}
    Action: finance
    Reward: -1.0



```python

```
