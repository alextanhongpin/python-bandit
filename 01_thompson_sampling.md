# Thompson sampling

From Wikipedia [^1]
> Thompson sampling,[1][2][3] named after William R. Thompson, is a heuristic for choosing actions that addresses the exploration-exploitation dilemma in the multi-armed bandit problem. It consists of choosing the action that maximizes the expected reward with respect to a randomly drawn belief.

[^1]: https://en.wikipedia.org/wiki/Thompson_sampling

# Resources
- https://gdmarmerola.github.io/ts-for-bernoulli-bandit/
- https://github.com/gdmarmerola/interactive-intro-rl/blob/master/notebooks/ts_for_multi_armed_bandit.ipynb


```python
import numpy as np
import matplotlib.pyplot as plt


class MAB:
    def __init__(self, bandit_probs):
        self.bandit_probs = bandit_probs

    def draw(self, k):
        reward = np.random.binomial(1, self.bandit_probs[k]) # Returns either 0 or 1
        regret = np.max(self.bandit_probs) - self.bandit_probs[k]
        return reward, regret
```


```python
class eGreedyPolicy:
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def choose_bandit(self, total_count, success_count, n_bandits):
        success_ratio = success_count/total_count
        
        best_action = np.argmax(success_ratio)
        if np.random.random() < self.epsilon:
            # Returning random action, excluding best
            return np.random.choice(list(range(n_bandits)) - best_action)
        else:
            # Returning best greedy action.
            return best_action
```


```python
# e-greedy policy
class UCBPolicy:
    
    # initializing
    def __init__(self):
        
        # nothing to do here
        pass
    
    # choice of bandit
    def choose_bandit(self, total_count, success_count, n_bandits):
        # ratio of sucesses vs total
        success_ratio = success_count/total_count
        
        # computing square root term
        sqrt_term = np.sqrt(2*np.log(np.sum(total_count))/total_count)
        
        # returning best greedy action
        return np.argmax(success_ratio + sqrt_term)    
```


```python
class TSPolicy:
    
    # initializing
    def __init__(self):
        
        # nothing to do here
        pass
    
    # choice of bandit
    def choose_bandit(self, total_count, success_count, n_bandits):
        # list of samples, for each bandit
        samples_list = []
        
        # sucesses and failures
        failure_count = total_count - success_count
                    
        # drawing a sample from each bandit distribution
        samples_list = [np.random.beta(1 + a, 1 + b) for a, b in zip(success_count, failure_count)]
                                
        # returning bandit with best sample
        return np.argmax(samples_list)    
```


```python
import numpy as np

# defining a set of bandits with known probabilites
bandit_probs = [0.35, 0.40, 0.30, 0.25]

# instance of our MAB class
mab = MAB(bandit_probs)

# policy
egreedy_policy = eGreedyPolicy(0.1)
ucb_policy = UCBPolicy()
ts_policy = TSPolicy()
```


```python
def random_policy(k_array, reward_array, N_BANDITS):
    return np.random.choice(range(N_BANDITS))
```


```python
N_DRAWS = 500
N_BANDITS = len(mab.bandit_probs)

policies = [random_policy, egreedy_policy.choose_bandit, ucb_policy.choose_bandit, ts_policy.choose_bandit]

for policy in policies:
    k_array = np.zeros(N_BANDITS)
    reward_array = np.zeros(N_BANDITS)
    total_regret = 0
    
    for i in range(N_DRAWS):
        k = policy(k_array, reward_array, N_BANDITS)
        reward, regret = mab.draw(k)
        k_array[k] += 1
        reward_array[k] += reward
        total_regret += regret
    print(k_array, reward_array, total_regret)
```

    [132. 125. 114. 129.] [52. 53. 35. 38.] 37.35
    [ 18. 426.  45.  11.] [  4. 166.  15.   2.] 7.049999999999995
    [130. 130. 162.  78.] [47. 47. 64. 21.] 34.40000000000013
    [330.  80.  16.  74.] [110.  25.   1.  20.] 29.200000000000166


    /var/folders/7m/74_ct3hx33d878n626w1wxyc0000gn/T/ipykernel_38567/1435375336.py:6: RuntimeWarning: invalid value encountered in divide
      success_ratio = success_count/total_count
    /var/folders/7m/74_ct3hx33d878n626w1wxyc0000gn/T/ipykernel_38567/1492610970.py:13: RuntimeWarning: invalid value encountered in divide
      success_ratio = success_count/total_count
    /var/folders/7m/74_ct3hx33d878n626w1wxyc0000gn/T/ipykernel_38567/1492610970.py:16: RuntimeWarning: divide by zero encountered in log
      sqrt_term = np.sqrt(2*np.log(np.sum(total_count))/total_count)
    /var/folders/7m/74_ct3hx33d878n626w1wxyc0000gn/T/ipykernel_38567/1492610970.py:16: RuntimeWarning: invalid value encountered in sqrt
      sqrt_term = np.sqrt(2*np.log(np.sum(total_count))/total_count)
    /var/folders/7m/74_ct3hx33d878n626w1wxyc0000gn/T/ipykernel_38567/1492610970.py:16: RuntimeWarning: invalid value encountered in divide
      sqrt_term = np.sqrt(2*np.log(np.sum(total_count))/total_count)
    /var/folders/7m/74_ct3hx33d878n626w1wxyc0000gn/T/ipykernel_38567/1492610970.py:16: RuntimeWarning: divide by zero encountered in divide
      sqrt_term = np.sqrt(2*np.log(np.sum(total_count))/total_count)



```python

```
