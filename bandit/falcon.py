
import numpy as np

import torch
from .base import BaseBandit
from sklearn.feature_extraction import FeatureHasher
from .environment import feature_interaction

class Falcon(BaseBandit):
    def __init__(
        self,
        /,
        n_arms,
        *,
        model=create_model(),
        preprocess=FeatureHasher(n_features, input_type="string", alternate_sign=True),
        epoch=20,
        tuning_parameter=0.1,
        confidence_parameter=1.0
        gamma=1.0 # Learning rate
    ):
        super().__init__(n_arms)
        self.gamma = gamma.
        self.tuning_parameter =  tuning_paramter
        self.epoch = epoch

    def adaptive_learning_rate(self):
        gamma = self.gamma
        tuning_parameter = self.tuning_parameter
        epoch = self.epoch
        confidence_parameter = self.confidence_parameter
        K = self.n_arms

        self.gamma = tuning_parameter * K * (epoch) / np.log(K * np.log(np.square(epoch) / confidence_parameter)
