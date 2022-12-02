#%%
import tensorflow as tf
from tensorflow import keras

import gym
import argparse
import numpy as np
import matplotlib.pyplot as plt

tf.keras.backend.set_floatx('float64')


parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--update_interval', type=int, default=5)
parser.add_argument('--actor_lr', type=float, default=0.0005)
parser.add_argument('--critic_lr', type=float, default=0.001)

args = parser.parse_args()

from agent.A2C import Actor,Critic

class Agent:
    def __init__(self, env):
        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.action_bound = self.env.action_space.high[0]
        self.std_bound = [1e-2, 1.0]
        self.epsilon = 1.0  # exploration rate
        self.actor = Actor(self.state_dim, self.action_dim,
                           self.action_bound, self.std_bound)
        self.critic = Critic(self.state_dim)

    def td_target(self, reward, next_state, done):
        if done:
            return reward
        v_value = self.critic.model.predict(
            np.reshape(next_state, [1, self.state_dim]).astype(np.int32))
        return np.reshape(reward + args.gamma * v_value[0], [1, 1])

    def advatnage(self, td_targets, baselines):
        return td_targets - baselines

    def list_to_batch(self, list):
        batch = list[0]
        for elem in list[1:]:
            batch = np.append(batch, elem, axis=0)
        return batch

    def train(self, max_episodes=1000):
        rewards_avg = []
        reward_history = []
        for ep in range(max_episodes):
            state_batch = []
            action_batch = []
            td_target_batch = []
            advatnage_batch = []
            episode_reward, done = 0, False

            state = self.env.reset()

            while not done:
                # self.env.render()
                action = self.actor.get_action(state)
                action = np.clip(action, -self.action_bound, self.action_bound)
                print(f'action is : {action}')

                next_state, reward, done, _ = self.env.step(action)

                state = np.reshape(state, [1, self.state_dim])
                action = np.reshape(action, [1, self.action_dim])
                next_state = np.reshape(next_state, [1, self.state_dim])
                reward = np.reshape(reward, [1, 1])

                td_target = self.td_target((reward+8)/8, next_state, done)
                advantage = self.advatnage(
                    td_target, self.critic.model.predict(state.astype(np.int32)))

                state_batch.append(state)
                action_batch.append(action)
                td_target_batch.append(td_target)
                advatnage_batch.append(advantage)

                if len(state_batch) >= args.update_interval or done:
                    states = self.list_to_batch(state_batch)
                    actions = self.list_to_batch(action_batch)
                    td_targets = self.list_to_batch(td_target_batch)
                    advantages = self.list_to_batch(advatnage_batch)

                    actor_loss = self.actor.train(states, actions, advantages)
                    critic_loss = self.critic.train(states, td_targets)

                    state_batch = []
                    action_batch = []
                    td_target_batch = []
                    advatnage_batch = []

                episode_reward += reward[0][0]
                state = next_state[0]
                reward_history.append(episode_reward)
            print('EP{} EpisodeReward={}'.format(ep, episode_reward))
            
                    
            if (ep+1) % 10 == 0:
                rewards_avg.append(np.average(reward_history))
                print(rewards_avg)
                reward_history.clear()

            if (ep+1) % 100 == 0:
                self.actor.model.save_weights(
                        "./saves/hedge{}-A3C_actor_1.2.h5".format(ep+1 + 1000))
                self.critic.model.save_weights(
                        "./saves/hedge{}-A3C_crtic_1.2.h5".format(ep+1 + 1000))

            if (ep+1) % 20 == 0:
                print(np.arange(0, (ep+1), 10))
                plt.plot(np.arange(0, (ep+1), 10), rewards_avg)
                plt.xlabel('Episode')
                plt.ylabel('Total Reward')
                plt.show()
                print("plot")