# Categorical Draw

What is categorical draw?

In simpler terms, a categorical distribution is a type of probability distribution that can be used when a random variable can fall into one of several different categories. Each category has its own separate probability. The categories don't have a specific order, but we often assign numbers to them just to make things easier. The probabilities for each category must be between 0 and 1, and if you add up all the probabilities, the total must be 1. This type of distribution is very flexible and can be used to model any situation where you have a fixed number of discrete outcomes.


```python
from collections import Counter

import numpy as np
```

The easiest way to implement categorical draw is to use `np.random.choice`, where you can specify the probability `p` of each categories.


```python
categories = ["apple", "banana", "kiwi"]
probabilities = [0.1, 0.3, 0.6]
n = 1000
Counter(np.random.choice(categories, n, p=probabilities))
```




    Counter({'kiwi': 594, 'banana': 309, 'apple': 97})



To sample a single value, we can just do this:


```python
np.random.choice(categories, p=probabilities)
```




    'banana'




```python
Counter(np.random.choice(["yes", "no"], 1000, p=[0.1, 0.9]))
```




    Counter({'no': 899, 'yes': 101})



This code is simulating a random process where an event can result in one of three outcomes: "apple", "banana", or "kiwi". The probabilities of these outcomes are 0.1, 0.3, and 0.6 respectively.

Here's a step-by-step breakdown:

1. It first creates a list of categories and their corresponding probabilities.
2. It then initializes a counter dictionary to keep track of the number of times each category is chosen.
3. The code then enters a loop that runs `n` times, where `n` the number of draws. Each iteration of the loop represents a single trial of the random process.
4. In each trial, it generates a random number `r` between 0 and 1.
5. It then goes through each category in order. For each category, it adds the category's probability to a running total `t`. If `t` exceeds `r`, it increments the counter for the current category and breaks out of the loop.
6. After all trials are complete, it returns the counter dictionary, which contains the number of times each category was chosen.

In essence, this code is a simple simulation of a categorical distribution.


```python
categories = ["apple", "banana", "kiwi"]
probabilities = [0.1, 0.3, 0.6]
counter = {cat: 0 for cat in categories}
for i in range(n):
    r = np.random.uniform()
    t = 0
    for i, p in enumerate(probabilities):
        t += p
        if t > r:
            counter[categories[i]] += 1
            break
counter
```




    {'apple': 94, 'banana': 293, 'kiwi': 613}



A smarter but more complicated way to get the result in one line is:


```python
categories = ["apple", "banana", "kiwi"]
probabilities = [0.1, 0.3, 0.6]
counter = {cat: 0 for cat in categories}
for i in range(n):
    r = np.random.uniform()
    counter[categories[np.argmax(np.cumsum(probabilities) > r)]] += 1
counter
```
