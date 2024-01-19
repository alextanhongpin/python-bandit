```python
import pandas as pd
import seaborn as sns

import bandit.environment as env
from bandit.neural_bandit import (
    EpsilonGreedyNeuralBandit,
    EpsilonGreedyNeuralPerArmBandit,
    SoftmaxNeuralBandit,
    SoftmaxNeuralPerArmBandit,
)

sns.set_theme()
```


```python
import importlib

import bandit

importlib.reload(bandit.neural_bandit)
```




    <module 'bandit.neural_bandit' from '/Users/alextanhongpin/Documents/python/python-bandit/bandit/neural_bandit.py'>




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
```


```python
N = 5_000
df = pd.DataFrame(index=range(N))
```


```python
model = ContextualEpsilonGreedyNeuralBandit(env.actions, epsilon=1.0)
total_rewards = 0
avg_rewards = []
for i, ctx in env.random_context(N):
    action = model.predict(ctx)
    reward = env.get_cost(ctx, action)
    model.fit(ctx, action, reward)
    total_rewards += max(0, reward)
    avg_rewards.append(total_rewards / (i + 1))
df[model.__name__] = avg_rewards
```

    100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5000/5000 [00:14<00:00, 339.74it/s]



```python
model = ContextualEpsilonGreedyNeuralBandit(env.actions, epsilon=0.9)
total_rewards = 0
avg_rewards = []
for i, ctx in env.random_context(N):
    action = model.predict(ctx)
    reward = env.get_cost(ctx, action)
    model.fit(ctx, action, reward)
    total_rewards += max(0, reward)
    avg_rewards.append(total_rewards / (i + 1))
df[model.__name__] = avg_rewards
```

    100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5000/5000 [00:12<00:00, 390.41it/s]



```python
model = ContextualSoftmaxNeuralBandit(env.actions, temperature=0.2)
total_rewards = 0
avg_rewards = []
for i, ctx in env.random_context(N):
    action = model.predict(ctx)
    reward = env.get_cost(ctx, action)
    model.fit(ctx, action, reward)
    total_rewards += max(0, reward)
    avg_rewards.append(total_rewards / (i + 1))
df[model.__name__] = avg_rewards
```

    100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5000/5000 [00:12<00:00, 394.69it/s]



```python
model = ContextualSoftmaxNeuralBandit(env.actions, temperature=0.5)
total_rewards = 0
avg_rewards = []
for i, ctx in env.random_context(N):
    action = model.predict(ctx)
    reward = env.get_cost(ctx, action)
    model.fit(ctx, action, reward)
    total_rewards += max(0, reward)
    avg_rewards.append(total_rewards / (i + 1))
df[model.__name__] = avg_rewards
```

    100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5000/5000 [00:14<00:00, 350.63it/s]



```python
model = ContextualEpsilonGreedyNeuralPerArmBandit(env.actions, epsilon=1.0)
total_rewards = 0
avg_rewards = []
for i, ctx in env.random_context(N):
    action = model.predict(ctx)
    reward = env.get_cost(ctx, action)
    model.fit(ctx, action, reward)
    total_rewards += max(0, reward)
    avg_rewards.append(total_rewards / (i + 1))
df[model.__name__] = avg_rewards
```

    100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5000/5000 [00:20<00:00, 242.60it/s]



```python
model = ContextualEpsilonGreedyNeuralPerArmBandit(env.actions, epsilon=0.9)
total_rewards = 0
avg_rewards = []
for i, ctx in env.random_context(N):
    action = model.predict(ctx)
    reward = env.get_cost(ctx, action)
    model.fit(ctx, action, reward)
    total_rewards += max(0, reward)
    avg_rewards.append(total_rewards / (i + 1))
df[model.__name__] = avg_rewards
```

    100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5000/5000 [00:22<00:00, 223.56it/s]



```python
model = ContextualSoftmaxNeuralPerArmBandit(env.actions, temperature=0.2)
total_rewards = 0
avg_rewards = []
for i, ctx in env.random_context(N):
    action = model.predict(ctx)
    reward = env.get_cost(ctx, action)
    model.fit(ctx, action, reward)
    total_rewards += max(0, reward)
    avg_rewards.append(total_rewards / (i + 1))
df[model.__name__] = avg_rewards
```

    100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5000/5000 [00:27<00:00, 180.71it/s]



```python
model = ContextualSoftmaxNeuralPerArmBandit(env.actions, temperature=0.5)
total_rewards = 0
avg_rewards = []
for i, ctx in env.random_context(N):
    action = model.predict(ctx)
    reward = env.get_cost(ctx, action)
    model.fit(ctx, action, reward)
    total_rewards += max(0, reward)
    avg_rewards.append(total_rewards / (i + 1))
df[model.__name__] = avg_rewards
```

    100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5000/5000 [00:22<00:00, 222.32it/s]



```python
import numpy as np

style = ["-", "--"]


def plot(*patterns):
    cols = []
    for col in df.columns:
        for pat in patterns:
            if pat not in col:
                break
        else:
            cols.append(col)
    repeat = (len(cols) + 1) // 2
    df[cols].plot(figsize=(12, 8), style=style * repeat)
```


```python
plot("Softmax")
```


    
![png](12_modular_files/12_modular_13_0.png)
    



```python
plot("SoftmaxNeuralBandit")
```


    
![png](12_modular_files/12_modular_14_0.png)
    



```python
plot("SoftmaxNeuralPerArmBandit")
```


    
![png](12_modular_files/12_modular_15_0.png)
    



```python
plot("Softmax", "0.2")
```


    
![png](12_modular_files/12_modular_16_0.png)
    



```python
plot("Softmax", "0.5")
```


    
![png](12_modular_files/12_modular_17_0.png)
    



```python
plot("Greedy")
```


    
![png](12_modular_files/12_modular_18_0.png)
    



```python
plot("GreedyNeuralBandit")
```


    
![png](12_modular_files/12_modular_19_0.png)
    



```python
plot("GreedyNeuralPerArmBandit")
```


    
![png](12_modular_files/12_modular_20_0.png)
    



```python
plot("Greedy", "0.9")
```


    
![png](12_modular_files/12_modular_21_0.png)
    



```python
plot("Greedy", "1.0")
```


    
![png](12_modular_files/12_modular_22_0.png)
    



```python
plot("GreedyNeural", "1.0")
```


    
![png](12_modular_files/12_modular_23_0.png)
    



```python
plot("NeuralBandit")
```


    
![png](12_modular_files/12_modular_24_0.png)
    



```python
plot("NeuralPerArmBandit")
```


    
![png](12_modular_files/12_modular_25_0.png)
    

