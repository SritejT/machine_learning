import numpy as np
from tensorflow import keras
import tensorflow as tf

class Normalizer(keras.layers.Layer):

    def __init__(self, epsilon=0.001, **kwargs):

        super().__init__(**kwargs)

        self.epsilon = epsilon

    def build(self, input_shape):

        self.alpha = self.add_weight(
            shape=input_shape[-1:],
            initializer="ones"
        )

        self.beta = self.add_weight(
            shape=input_shape[-1:],
            initializer="zeros"
        )

        super().build(input_shape)

    def call(self, X):

        mean, variance = tf.nn.moments(X, axes=-1, keepdims=True)

        return self.alpha * (X - mean) / (tf.sqrt(variance) + self.epsilon) + self.beta

    def get_config(self):

        base_config = super().get_config()

        return {**base_config, "eps": self.eps}
