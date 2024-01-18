```python
import numpy as np
import tensorflow as tf


class ContextualBandit:
    def __init__(self, num_actions, state_dim):
        self.num_actions = num_actions
        self.state_dim = state_dim
        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Dense(
                    32, activation="relu", input_shape=(self.state_dim,)
                ),
                tf.keras.layers.Dense(self.num_actions),
            ]
        )
        model.compile(optimizer="adam", loss="mse")
        return model

    def get_action(self, state):
        logits = self.model.predict(state)
        action = np.argmax(logits, axis=1)
        return action

    def train(self, states, actions, rewards):
        self.model.fit(states, actions, sample_weight=rewards, epochs=1, verbose=0)


# Usage:
bandit = ContextualBandit(num_actions=10, state_dim=5)

# Get an action for a given state
state = np.random.rand(1, 5)
action = bandit.get_action(state)

# Train the model with some states, actions and rewards
states = np.random.rand(10, 5)
actions = np.random.randint(0, 10, size=(10, 1))
rewards = np.random.rand(10, 1)
bandit.train(states, actions, rewards)
```
