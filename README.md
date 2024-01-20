# vowpal wabbit


## Installation


If you hit this error:
```
       Could NOT find Boost (missing: Boost_INCLUDE_DIR)
      Call Stack (most recent call first):
```

Install this:
```bash
$ brew install boost-python3
```

Then reinstall vowpalwabbit:
```bash
$ pip install vowpalwabbit
```

Seems like vowpalwabbit is not compatible with python 3.11 yet:

```bash
$ poetry env use 3.10
```

## Training

```bash
(python-bandit-py3.10) ➜  python-bandit python -m vowpalwabbit house_dataset
using no cache
Reading datafile = house_dataset
num sources = 1
Num weight bits = 18
learning rate = 0.5
initial_t = 0
power_t = 0.5
Enabled learners: gd, scorer-identity, count_label
Input label = SIMPLE
Output pred = SCALAR
average  since         example        example        current        current  current
loss     last          counter         weight          label        predict features

finished run
number of examples = 0
weighted example sum = 0.000000
weighted label sum = 0.000000
average loss = n.a.
total feature number = 0
```


## References

- https://gdmarmerola.github.io/ts-for-contextual-bandits/
- https://renan-cunha.github.io/categories/contextual-bandits/
- LinUCB: https://arxiv.org/pdf/1003.0146.pdf
- The implementation from the paper above: https://github.com/kfoofw/bandit_simulations/tree/master
- https://towardsdatascience.com/recommender-systems-using-linucb-a-contextual-multi-armed-bandit-approach-35a6f0eb6c4?gi=61d0618494e6
- http://auai.org/uai2017/proceedings/papers/171.pdf