```python
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm.autonotebook import tqdm

import bandit.environment as env
from bandit.policy import EGreedy, Softmax
from bandit.torch import NeuralBandit, NeuralPerArmBandit, create_model

sns.set_theme()
```

    /var/folders/7m/74_ct3hx33d878n626w1wxyc0000gn/T/ipykernel_90415/2096907171.py:8: TqdmExperimentalWarning: Using `tqdm.autonotebook.tqdm` in notebook mode. Use `tqdm.tqdm` instead to force console mode (e.g. in jupyter console)
      from tqdm.autonotebook import tqdm



```python
N = 500
```


```python
def run_simulation(bandit, policy=EGreedy(epsilon=0.0), n=N, dynamic=False):
    total_reward = 0
    avg_rewards = []
    rng = np.random.RandomState(42)

    for i in tqdm(range(n), disable=False):
        state = env.observe(rng)

        # 1. Predict the action.
        rewards = bandit.pull(state)

        action = policy(rewards)

        # 2. Act and get the reward.
        if dynamic and i > n // 2:
            get_cost = env.get_cost_new
        else:
            get_cost = env.get_cost
        reward = get_cost(state, env.actions[action])
        # Change reward to 0 or 1 instead of -1 or 1

        # 3. Update the model.
        bandit.update(state, action, reward)

        # 4. Save the reward.
        total_reward += max(0, reward)
        avg_rewards.append(total_reward / (i + 1))
    return avg_rewards, total_reward
```


```python
bandit = NeuralBandit(n_arms=len(env.actions))
avg_rewards, total_reward = run_simulation(bandit)
total_reward
```


      0%|          | 0/500 [00:00<?, ?it/s]





    346.0




```python
plt.plot(range(N), avg_rewards)
```




    [<matplotlib.lines.Line2D at 0x120ed4c70>]




    
![png](15_torch_bandit_files/15_torch_bandit_4_1.png)
    



```python
bandit = NeuralBandit(n_arms=len(env.actions))
avg_rewards, total_reward = run_simulation(bandit, dynamic=True)
total_reward
```


      0%|          | 0/500 [00:00<?, ?it/s]





    170.0




```python
plt.plot(range(N), avg_rewards)
```




    [<matplotlib.lines.Line2D at 0x1238fde10>]




    
![png](15_torch_bandit_files/15_torch_bandit_6_1.png)
    


## Neural Per Arm Bandit - Static


```python
bandit = NeuralPerArmBandit()
avg_rewards, total_reward = run_simulation(bandit)
total_reward
```


      0%|          | 0/500 [00:00<?, ?it/s]





    348.0




```python
plt.plot(range(N), avg_rewards)
```




    [<matplotlib.lines.Line2D at 0x12397b5e0>]




    
![png](15_torch_bandit_files/15_torch_bandit_9_1.png)
    


## Neural Per Arm Bandit - Dynamic


```python
bandit = NeuralPerArmBandit()
avg_rewards, total_reward = run_simulation(bandit, dynamic=True)
total_reward
```


      0%|          | 0/500 [00:00<?, ?it/s]





    470.0




```python
plt.plot(range(N), avg_rewards)
```




    [<matplotlib.lines.Line2D at 0x123a09e70>]




    
![png](15_torch_bandit_files/15_torch_bandit_12_1.png)
    



```python
bandit = NeuralBandit(n_arms=len(env.actions))

# bandit = NeuralPerArmBandit()

policy = EGreedy(epsilon=0.0)
rng = np.random.RandomState(42)
state = env.observe(rng)


rewards = bandit.pull(state)
print("rewards", rewards)

action = policy(rewards)

reward = env.get_cost(state, env.actions[action])
print("action", action)
print("reward", reward)

# 3. Update the model.

bandit.update(state, action, reward)
```

    rewards [-0.06754722  0.2109358  -0.13107799  0.06537426 -0.08250759 -0.00643894
     -0.08250759]
    action 1
    reward -1.0
