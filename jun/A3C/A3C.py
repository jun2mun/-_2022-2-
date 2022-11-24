#%%
import tensorflow as tf
from tensorflow import keras

import gym
import argparse
import numpy as np
from threading import Thread
from multiprocessing import cpu_count
import matplotlib.pyplot  as plt
tf.keras.backend.set_floatx('float64')

parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--update_interval', type=int, default=5)
parser.add_argument('--actor_lr', type=float, default=0.0005)
parser.add_argument('--critic_lr', type=float, default=0.001)

args = parser.parse_args()

CUR_EPISODE = 0

class Actor:
    def __init__(self, state_dim, action_dim, action_bound, std_bound):
        self.state_dim = state_dim # observation 개수
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.std_bound = std_bound
        self.model = self.create_model()
        self.opt = tf.keras.optimizers.Adam(args.actor_lr)
        self.entropy_beta = 0.01

    def create_model(self):
        state_input = keras.layers.Input((self.state_dim,))
        dense_1 = keras.layers.Dense(32, activation='relu')(state_input)
        dense_2 = keras.layers.Dense(32, activation='relu')(dense_1)
        out_mu = keras.layers.Dense(self.action_dim, activation='tanh')(dense_2)
        mu_output = keras.layers.Lambda(lambda x: x * self.action_bound)(out_mu)
        std_output = keras.layers.Dense(self.action_dim, activation='softplus')(dense_2)
        return tf.keras.models.Model(state_input, [mu_output, std_output])

    def get_action(self, state): # act 함수
        state = np.reshape(state, [1, self.state_dim])
        mu, std = self.model.predict(state)
        mu, std = mu[0], std[0]
        return np.random.normal(mu, std, size=self.action_dim) # (mu ~ std)

    def log_pdf(self, mu, std, action): # pdf 함수
        std = tf.clip_by_value(std, self.std_bound[0], self.std_bound[1]) # std값을 bound[0] bound[1] 범위 안으로 만듬.
        var = std ** 2
        log_policy_pdf = -0.5 * (action - mu) ** 2 / \
            var - 0.5 * tf.math.log(var * 2 * np.pi)
        return tf.reduce_sum(log_policy_pdf, 1, keepdims=True)

    def compute_loss(self, mu, std, actions, advantages): # loss 함수
        log_policy_pdf = self.log_pdf(mu, std, actions)
        loss_policy = log_policy_pdf * advantages
        return tf.reduce_sum(-loss_policy)

    def train(self, states, actions, advantages):
        with tf.GradientTape() as tape:
            mu, std = self.model(states, training=True)
            loss = self.compute_loss(mu, std, actions, advantages)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

class Critic:
    def __init__(self, state_dim):
        self.state_dim = state_dim
        self.model = self.create_model()
        self.opt = tf.keras.optimizers.Adam(args.critic_lr)

    def create_model(self):
        return tf.keras.Sequential([
            keras.layers.Input((self.state_dim,)),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(1, activation='linear')
        ])

    def compute_loss(self, v_pred, td_targets):
        mse = tf.keras.losses.MeanSquaredError()
        return mse(td_targets, v_pred)

    def train(self, states, td_targets):
        with tf.GradientTape() as tape:
            v_pred = self.model(states, training=True)
            assert v_pred.shape == td_targets.shape
            loss = self.compute_loss(v_pred, tf.stop_gradient(td_targets))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

class Agent:
    def __init__(self, env):
        #env = gym.make(env_name)
        self.env= env
        self.env_name = 'env_name'#env_name
        self.state_dim = env.observation_space.shape[0] # observation 개수
        self.action_dim = env.action_space.shape[0] # action 개수
        self.action_bound = env.action_space.high[0] # action 상한선
        self.std_bound = [1e-2, 1.0] # ??

        self.global_actor = Actor(
            self.state_dim, self.action_dim, self.action_bound, self.std_bound)
        self.global_critic = Critic(self.state_dim)
        self.num_workers = cpu_count()

    def train(self, max_episodes=100000):
        workers = []
        global id
        id = 0

        env = self.env
        env.setid(1)
        worker =WorkerAgent(env, self.global_actor, self.global_critic, max_episodes,id)
        worker.start()
        '''
        for i in range(self.num_workers):
            #env = gym.make(self.env_name)
            env = self.env
            env.setid(i)
            id +=1
            workers.append(WorkerAgent(
                env, self.global_actor, self.global_critic, max_episodes,id))

        for worker in workers:
            worker.start()

        for worker in workers:
            worker.join()
        '''

class WorkerAgent(Thread):
    def __init__(self, env, global_actor, global_critic, max_episodes,id):
        Thread.__init__(self)
        self.id = id
        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.action_bound = self.env.action_space.high[0]
        self.std_bound = [1e-2, 1.0]

        self.max_episodes = max_episodes
        self.global_actor = global_actor
        self.global_critic = global_critic
        self.actor = Actor(self.state_dim, self.action_dim,
                           self.action_bound, self.std_bound)
        self.critic = Critic(self.state_dim)

        #self.actor.model.load_weights('C:\\Users\\owner\\Desktop\\경희대학교\\2022-2학기\\데캡톤\\프로젝트\\jun\\save\\hedge11000-A3C_actor_1.2.h5')
        #self.critic.model.load_weights('C:\\Users\\owner\\Desktop\\경희대학교\\2022-2학기\\데캡톤\\프로젝트\\jun\\save\\hedge11000-A3C_crtic_1.2.h5')

        self.update_target_model()
        
    def update_target_model(self):
        self.actor.model.set_weights(self.global_actor.model.get_weights())
        self.critic.model.set_weights(self.global_critic.model.get_weights())

        #print(f'global_actor model weights : {self.global_actor.model.get_weights()}')
        #print(f'global_crtici model weights : {self.global_critic.model.get_weights()}')

    def n_step_td_target(self, rewards, next_v_value, done):
        td_targets = np.zeros_like(rewards)
        cumulative = 0
        if not done:
            cumulative = next_v_value

        for k in reversed(range(0, len(rewards))):
            cumulative = args.gamma * cumulative + rewards[k]
            td_targets[k] = cumulative
        return td_targets

    def advatnage(self, td_targets, baselines):
        return td_targets - baselines

    def list_to_batch(self, list):
        batch = list[0]
        for elem in list[1:]:
            batch = np.append(batch, elem, axis=0)
        return batch

    def train(self):
        CUR_EPISODE = 0
        rewards_avg = []
        reward_history = []

        #while self.max_episodes >= CUR_EPISODE:
        for CUR_EPISODE in range(self.max_episodes):
            state_batch = []
            action_batch = []
            reward_batch = []
            episode_reward, done = 0, False
            state = self.env.reset()

            #while not done:
            MAX_EP_STEP = 30
            for time in range(MAX_EP_STEP):
                print(f'======= my id is : {self.id} and cur step is : {CUR_EPISODE} =======')
                
                action = self.actor.get_action(state) # 1. action
                print(f'my action is {action}')
                #action = np.clip(action, -self.action_bound, self.action_bound)
                action = np.clip(action, 0, self.action_bound)


                next_state, reward, done, _ = self.env.step(action) # 2. step 

                state = np.reshape(state, [1, self.state_dim])
                action = np.reshape(action, [1, 1])
                next_state = np.reshape(next_state, [1, self.state_dim])
                reward = np.reshape(reward, [1, 1])

                state_batch.append(state)
                action_batch.append(action)
                reward_batch.append(reward)

                episode_reward += reward[0][0]

                #if time == MAX_EP_STEP -1 :
                #    done = True
                state = next_state[0]


                if len(state_batch) >= args.update_interval or done:

                    states = self.list_to_batch(state_batch)
                    actions = self.list_to_batch(action_batch)
                    rewards = self.list_to_batch(reward_batch)

                    next_v_value = self.critic.model.predict(next_state.astype(np.float32)) # (1,1)
                    #print(f' v_value = {next_v_value}')
                    td_targets = self.n_step_td_target(
                        (rewards+8)/8, next_v_value, done)
                    advantages = td_targets - self.critic.model.predict(states)
                    
                    actor_loss = self.global_actor.train(
                        states, actions, advantages)
                    critic_loss = self.global_critic.train(
                        states, td_targets)
                    #print(f'{td_targets} {advantages} {actor_loss}')
                    self.update_target_model() # 가중치 업데이트
                    
                    state_batch = []
                    action_batch = []
                    reward_batch = []
                    td_target_batch = []
                    advatnage_batch = []
                    
                    print('ID {} EP{} EpisodeReward={}'.format(self.id,CUR_EPISODE, episode_reward))
                    reward_history.append(episode_reward)
                    break
                
                    
            if (CUR_EPISODE+1) % 10 == 0:
                rewards_avg.append(np.average(reward_history))
                print(rewards_avg)
                reward_history.clear()

            if (CUR_EPISODE+1) % 1000 == 0:
                self.actor.model.save_weights(
                        "./jun/save/hedge{}-A3C_actor_1.2.h5".format(CUR_EPISODE+1 + 1000))
                self.critic.model.save_weights(
                        "./jun/save/hedge{}-A3C_crtic_1.2.h5".format(CUR_EPISODE+1 + 1000))

            if (CUR_EPISODE+1) % 50 == 0:
                print(np.arange(0, (CUR_EPISODE+1), 10))
                plt.plot(np.arange(0, (CUR_EPISODE+1), 10), rewards_avg)
                plt.xlabel('Episode')
                plt.ylabel('Total Reward')
                plt.show()

    def save(self,filepath):
        self.actor.save(filepath) # Actor 가중치 저장
        self.critic.save(filepath) # Critic 가중치 저장

    def load(self,filepath):
        self.actor.load(filepath) # Actor 가중치 저장
        self.critic.load(filepath) # Critic 가중치 저장

    def run(self): # Thread 실행하기 위한 run 함수(thread.start())
        self.train()


