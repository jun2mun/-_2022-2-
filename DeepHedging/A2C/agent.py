
import numpy as np
from keras.layers import Dense, Input, Lambda, LSTM
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
from collections import deque
import random


class A2CAgent:
    def __init__(self, state_size, state_seq_length, action_size):
        # self.render = False # if you want to see Cartpole learning, then change to True

        self.state_size = state_size
        self.state_seq_length = state_seq_length
        self.action_size = action_size
        self.value_size = 1
        self.exp_replay = deque(maxlen=2000)

        # get gym environment name
        # these are hyper parameters for the A3C
        self.actor_lr = 0.0001
        self.critic_lr = 0.001
        self.discount_factor = .9

        # create model for actor and critic network
        self.actor, self.critic = self.build_model()

        # method for training actor and critic network
        # self.optimizer = [self.actor_optimizer(), self.critic_optimizer()]
        self.optimize_actor = self.actor_optimizer()  # 5
        self.optimize_critic = self.critic_optimizer()

    def build_model(self):
        state = Input(batch_shape=(None, self.state_seq_length, self.state_size))

        x = LSTM(120, return_sequences=True)(state)
        x = LSTM(100)(x)

        actor_input = Dense(100, activation='relu', kernel_initializer='he_uniform')(x)
        # actor_hidden = Dense(self.hidden2, activation='relu')(actor_input)
        mu = Dense(self.action_size, activation='tanh', kernel_initializer='he_uniform')(actor_input)
        sigma_0 = Dense(self.action_size, activation='softplus', kernel_initializer='he_uniform')(actor_input)
        sigma = Lambda(lambda x: x + 0.0001)(sigma_0)

        critic_input = Dense(30, activation='relu', kernel_initializer='he_uniform')(x)
        # value_hidden = Dense(self.hidden2, activation='relu')(critic_input)
        state_value = Dense(1, activation='linear', kernel_initializer='he_uniform')(critic_input)

        actor = Model(inputs=state, outputs=(mu, sigma))
        critic = Model(inputs=state, outputs=state_value)

        actor.make_predict_function()
        critic.make_predict_function()

        actor.summary()
        critic.summary()

        return actor, critic

    def actor_optimizer(self):
        action = K.placeholder(shape=(None, 1))
        advantages = K.placeholder(shape=(None, 1))

        # mu = K.placeholder(shape=(None, self.action_size))
        # sigma_sq = K.placeholder(shape=(None, self.action_size))

        mu, sigma_sq = self.actor.output

        pdf = 1. / K.sqrt(2. * np.pi * sigma_sq) * K.exp(-K.square(action - mu) / (2. * sigma_sq))
        log_pdf = K.log(pdf + K.epsilon())
        entropy = K.sum(0.5 * (K.log(2. * np.pi * sigma_sq) + 1.))

        exp_v = log_pdf * advantages

        exp_v = K.sum(exp_v + 0.01 * entropy)
        actor_loss = -exp_v

        optimizer = Adam(lr=self.actor_lr)
        updates = optimizer.get_updates(actor_loss, self.actor.trainable_weights)

        train = K.function([self.actor.input, action, advantages], [], updates=updates)
        return train

    # make loss function for Value approximation
    def critic_optimizer(self):
        discounted_reward = K.placeholder(shape=(None, 1))

        value = self.critic.output

        loss = K.mean(K.square(discounted_reward - value))

        optimizer = Adam(lr=self.critic_lr)
        updates = optimizer.get_updates(loss, self.critic.trainable_weights)
        train = K.function([self.critic.input, discounted_reward], [], updates=updates)
        return train

    # using the output of policy network, pick action stochastically
    def get_action(self, state):
        mu, sigma_sq = self.actor.predict(np.reshape(state, [1, self.state_seq_length, self.state_size]))
        # sigma_sq = np.log(np.exp(sigma_sq + 1))
        epsilon = np.random.randn(self.action_size)
        # action = norm.rvs(loc=mu, scale=sigma_sq,size=1)
        action = mu + np.sqrt(sigma_sq) * epsilon
        action = np.clip(action, -2, 2)
        return action

    # update policy network every episode
    def train_model(self, state, action, reward, next_state, done):
        self.exp_replay.append((state, action, reward, next_state, done))

        (state, action, reward, next_state, done) = random.sample(self.exp_replay, 1)[0]

        target = np.zeros((1, self.value_size))
        advantages = np.zeros((1, self.action_size))

        value = self.critic.predict(state)[0]
        next_value = self.critic.predict(next_state)[0]

        if done:
            advantages[0] = reward - value
            target[0][0] = reward
        else:
            advantages[0] = reward + self.discount_factor * (next_value) - value
            target[0][0] = reward + self.discount_factor * next_value

        self.optimize_actor([state, action, advantages])
        self.optimize_critic([state, target])