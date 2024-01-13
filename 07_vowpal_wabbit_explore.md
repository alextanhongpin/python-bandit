```python
# !poetry run python -m vowpalwabbit -d train.dat --cb_explore 4 --bag 5
import vowpalwabbit

vw = vowpalwabbit.Workspace("--cb_explore 4 --bag 5", quiet=False)
```

    using no cache
    Reading datafile = none
    num sources = 0
    Num weight bits = 18
    learning rate = 0.5
    initial_t = 0
    power_t = 0.5
    cb_type = mtr
    Enabled learners: gd, scorer-identity, csoaa_ldf-rank, cb_adf, cb_explore_adf_bag, shared_feature_merger, cb_to_cbadf
    Input label = CB
    Output pred = ACTION_PROBS
    average  since         example        example        current        current  current
    loss     last          counter         weight          label        predict features



```python
use_train = False
if use_train:
    with open("train.dat") as f:
        for line in f:
            line = line.strip()
            print(line)
            vw.learn(line)
```


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
def sample_custom_pmf(pmf):
    action = np.argmax(pmf)
    return action, pmf[action]
```


```python
import random

import numpy as np

while False:
    pmf = vw.predict("| a b c")
    action, probability = sample_custom_pmf(pmf)
    print("showing action:", action + 1)
    value = input("enter y, n or q:")
    if value == "q":
        break
    # Cost must be -tive for positive rewards.
    cost = -1 if value == "y" else 1
    # Action index starts from 1
    vw.learn(f"{action+1}:{cost}:{probability} | a b c")
    print(f"{action+1}:{cost}:{probability} | a b c")
```


```python
import random
from collections import defaultdict

import numpy as np

cost_sum = 0.0
ctr = []
success = defaultdict(int)


for i in range(1, 1000):
    pmf = vw.predict("| a b c")
    action, probability = sample_custom_pmf(pmf)
    # Cost must be +tive for positive rewards.
    cost = np.random.binomial(1, p=probability)
    # Action index starts from 1
    success[action] += 1
    vw.learn(f"{action+1}:{cost}:{probability} | a b c")
    # print(f"{action+1}:{cost}:{probability} | a b c")
    cost_sum += cost
    ctr.append(cost_sum / i)
```

    n.a.     n.a.                1            1.0        unknown         0:0.25       16
    0.000000 0.000000            2            2.0       0:0:0.25         0:0.25       16
    0.000000 0.000000            4            4.0       0:0:0.25         0:0.25       16
    0.250000 0.500000            8            8.0       0:1:0.25         0:0.25       16
    0.125000 0.000000           16           16.0        1:0:0.3          1:0.3       16
    0.187500 0.250000           32           32.0       2:1:0.42         2:0.42       16
    0.343750 0.500000           64           64.0        0:1:0.4          1:0.4       16
    0.546875 0.750000          128          128.0        0:0:0.4          0:0.4       16
    0.625000 0.703125          256          256.0          0:1:1            0:1       16
    0.691406 0.757813          512          512.0        0:0:0.6          0:0.6       16
    0.728516 0.765625         1024         1024.0        0:1:0.8          0:0.8       16



```python
import matplotlib.pyplot as plt

plt.plot(range(1, 1000), ctr)
```




    [<matplotlib.lines.Line2D at 0x112f69180>]




    
![png](07_vowpal_wabbit_explore_files/07_vowpal_wabbit_explore_6_1.png)
    



```python
success
```




    defaultdict(int, {0: 533, 1: 233, 2: 26, 3: 207})




```python
cost_sum
```




    754.0


