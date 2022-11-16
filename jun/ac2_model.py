import os
import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization, Dropout
from keras.optimizers import RMSprop

class a2c_Model:
    def __init__(self, observation_space, num_actions):

        input = Input(shape=(observation_space,),name='policy_input')
        
        #model's policy branch
        policy = Dense(32, activation="relu")(input)
        policy = Dense(16, activation="relu")(policy)
        logits = Dense(num_actions)(policy)

        #model's value function branch
        value_fn = Dense(32, activation="relu")(input)
        value_fn = Dense(16, activation="relu")(value_fn)
        value_fn = Dense(1)(value_fn)

        #defining the full model
        self.network = Model(inputs=input, outputs=[logits, value_fn])

    def forward_pass(self, inputs):
        #function to get forward pass
        x = tf.convert_to_tensor(inputs)
        return self.network(x)

    def actionfromdistribution(self, logits):
        #function to get a particular action from the logits of the different actions
        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)

    def action_value(self, observation):
        #function to return:
        #what action to take next
        #value function
        #based on input observation
        logits, value = self.forward_pass(observation)
        action = self.actionfromdistribution(logits)
        return np.squeeze(action, axis=-1), np.squeeze(value, axis=-1)


if __name__=="__main__":
    env = gym.make('CartPole-v0')
    models = a2c_Model(env.observation_space.shape[0], env.action_space.n)
    obs = env.reset()
    obs = np.expand_dims(obs, axis=0)
    print(models.action_value(obs))