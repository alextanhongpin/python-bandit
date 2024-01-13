# Contextual Bandit using Vowpal Wabbit

https://vowpalwabbit.org/docs/vowpal_wabbit/python/latest/tutorials/python_Contextual_bandits_and_Vowpal_Wabbit.html


```python
import pandas as pd
```


```python
train_data = [
    {
        "action": 1,
        "cost": 2,
        "probability": 0.4,
        "feature1": "a",
        "feature2": "c",
        "feature3": "",
    },
    {
        "action": 3,
        "cost": 0,
        "probability": 0.2,
        "feature1": "b",
        "feature2": "d",
        "feature3": "",
    },
    {
        "action": 4,
        "cost": 1,
        "probability": 0.5,
        "feature1": "a",
        "feature2": "b",
        "feature3": "",
    },
    {
        "action": 2,
        "cost": 1,
        "probability": 0.3,
        "feature1": "a",
        "feature2": "b",
        "feature3": "c",
    },
    {
        "action": 3,
        "cost": 1,
        "probability": 0.7,
        "feature1": "a",
        "feature2": "d",
        "feature3": "",
    },
]

train_df = pd.DataFrame(train_data)

# Add index to data frame
train_df["index"] = range(1, len(train_df) + 1)
train_df = train_df.set_index("index")
train_df
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
      <th>action</th>
      <th>cost</th>
      <th>probability</th>
      <th>feature1</th>
      <th>feature2</th>
      <th>feature3</th>
    </tr>
    <tr>
      <th>index</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2</td>
      <td>0.4</td>
      <td>a</td>
      <td>c</td>
      <td></td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0</td>
      <td>0.2</td>
      <td>b</td>
      <td>d</td>
      <td></td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>0.5</td>
      <td>a</td>
      <td>b</td>
      <td></td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>1</td>
      <td>0.3</td>
      <td>a</td>
      <td>b</td>
      <td>c</td>
    </tr>
    <tr>
      <th>5</th>
      <td>3</td>
      <td>1</td>
      <td>0.7</td>
      <td>a</td>
      <td>d</td>
      <td></td>
    </tr>
  </tbody>
</table>
</div>




```python
test_data = [
    {"feature1": "b", "feature2": "c", "feature3": ""},
    {"feature1": "a", "feature2": "", "feature3": "b"},
    {"feature1": "b", "feature2": "b", "feature3": ""},
    {"feature1": "a", "feature2": "", "feature3": "b"},
]

test_df = pd.DataFrame(test_data)

# Add index to data frame
test_df["index"] = range(1, len(test_df) + 1)
test_df = test_df.set_index("index")
test_df
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
      <th>feature1</th>
      <th>feature2</th>
      <th>feature3</th>
    </tr>
    <tr>
      <th>index</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>c</td>
      <td></td>
    </tr>
    <tr>
      <th>2</th>
      <td>a</td>
      <td></td>
      <td>b</td>
    </tr>
    <tr>
      <th>3</th>
      <td>b</td>
      <td>b</td>
      <td></td>
    </tr>
    <tr>
      <th>4</th>
      <td>a</td>
      <td></td>
      <td>b</td>
    </tr>
  </tbody>
</table>
</div>




```python
import vowpalwabbit

vw = vowpalwabbit.Workspace("--cb 4", quiet=False)
```

    using no cache
    Reading datafile = none
    num sources = 0
    Num weight bits = 18
    learning rate = 0.5
    initial_t = 0
    power_t = 0.5
    cb_type = mtr
    Enabled learners: gd, scorer-identity, csoaa_ldf-rank, cb_adf, shared_feature_merger, cb_to_cbadf
    Input label = CB
    Output pred = MULTICLASS
    average  since         example        example        current        current  current
    loss     last          counter         weight          label        predict features



```python
for i in train_df.index:
    action, cost, probability, feature1, feature2, feature3 = train_df.loc[i]
    
    # Construct the example in the required vw format.
    learn_example = f'{action}:{cost}:{probability} | {feature1} {feature2} {feature3}'
    
    # Here we do the actual learning.
    vw.learn(learn_example)
```

    5.000000 5.000000            1            1.0        0:2:0.4            0:0       12
    2.500000 0.000000            2            2.0        2:0:0.2            1:0       12
    2.083333 1.666667            4            4.0        1:1:0.3            1:0       16



```python
for j in test_df.index:
    feature1, feature2, feature3 = test_df.loc[j]
    test_example = f'| {feature1} {feature2} {feature3}'
    choice = vw.predict(test_example)
    print(j, choice)
```

    1 3
    2 3
    3 3
    4 3


    1.952381 1.428571            8            8.0        unknown         2:0.13       12



```python
vw.save("cb.model")
del vw

vw = vowpalwabbit.Workspace("--cb 4 -i cb.model", quiet=True)
print(vw.predict("| a b"))
```

    3


    
    finished run
    number of examples = 9
    weighted example sum = 9.000000
    weighted label sum = 0.000000
    average loss = 1.952381
    total feature number = 112



```python
print(vw.predict("| a"))
```

    3



```python
print(vw.predict("| b"))
```

    3



```python
print(vw.predict("| a b c"))
```

    3



```python
print(vw.predict("| a d"))
```

    4



```python
while True:
    n = input('enter a number between 1 to 4:') or '1'
    action, cost, probability = int(n), 1, 0.25
    if action < 0:
        break
    # Construct the example in the required vw format.
    learn_example = f'{action}:{cost}:{probability} | a b'
    
    # Here we do the actual learning.
    vw.learn(learn_example)

    print(vw.predict("| a b"))
```

    enter a number between 1 to 4: 1


    3


    enter a number between 1 to 4: -1



```python

```
