import numpy as np
import tensorflow as tf
from tensorflow import keras
from game_env import GameEnv
from tf_agents.environments.wrappers import ActionDiscretizeWrapper
from tf_agents.environments.tf_py_environment import TFPyEnvironment

env = GameEnv()
env = ActionDiscretizeWrapper(env, num_actions=4)
tf_env = TFPyEnvironment(env)

model = keras.models.Sequential([
    keras.layers.Dense(10, activation="elu", input_shape=[4]),
    keras.layers.Dense(1, activation="sigmoid"),
])

def play_one_step(env, obs, model, loss_fn):
    with tf.GradientTape() as tape:
        left_proba = model(obs[np.newaxis])
        action = (tf.random.uniform([1, 1]) > left_proba)
        y_target = tf.constant([[1.]]) - tf.cast(action, tf.float32)
        loss = tf.reduce_mean(loss_fn(y_target, left_proba))
    grads = tape.gradient(loss, model.trainable_variables)
    obs, reward, done, info = env.step(int(action[0, 0].numpy()))
    return obs, reward, done, grads

