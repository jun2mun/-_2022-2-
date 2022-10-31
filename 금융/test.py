from keras.layers import Dense, Activation, Input
from keras.models import Model,load_model
from keras.optimizers import Adam
import keras.backend as K
import numpy as np
import tensorflow as tf

class Agent(object):
    def __init__(self,ALPHA,GAMMA=0.99,n_actions=4,
                layer1_size=16,layer2_size=16,input_dims=128,
                fname ='reinforce.h5'):
        self.gamma = GAMMA
        self.lr = ALPHA
        self.G = 0
        self.input_dims = input_dims
        self.fc1_dims = layer1_size
        self.fc2_dims = layer2_size
        self.n_actions = n_actions
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []

        self.policy, self.predict = self.build_policy_network(ALPHA,n_actions,
                                            input_dims,layer1_size,layer2_size)
        self.action_space = [i for i in range(n_actions)]
        self.model_file = fname


    def build_policy_network(self):
        input = tf.keras.layers.Input(shape=(30,1)) # (time_steps,feature)
        # shape=(self.input_dims)
        output,state_h,state_c = tf.keras.layers.LSTM(64,return_state=True)(input)
        output = tf.keras.layers.Dense(32,activation='tanh')(output)
        # self.fc1_dims
        # self.fc2.dims
        output = tf.keras.layers.Dense(1)(output)
        # probs = Dense(self.n_actions, activation = 'softmax)(위)

        def custom_loss(y_true, y_pred):
            out = K.clip(y_pred,1e-8,1-1e-8) # height 설정 out의
            log_lik = y_true*K.log(out)

            return K.sum(-log_lik*advantages)

        strategy = tf.reshape(output,(30,n_sims))
        strategy = tf.cast(strategy,tf.float64)
        option = tf.maximum(STOCK_T - STOCK_K,0)
        Hedging_PnL = - option + tf.reduce_sum(STOCK_dS * strategy, axis=0)
        Hedging_PnL_paths = - option + STOCK_dS * strategy #(10000,) + (30,10000)
        CVaR, idx = tf.math.top_k(-Hedging_PnL,k=tf.cast((1-alpha)* n_sims, tf.int32)) # must set to int32 !!
        CVaR = tf.reduce_mean(-Hedging_PnL)
        policy = tf.keras.Model(inputs=input,outputs=CVaR)
        policy.compile(optimizer=Adam(lr=self.lr),loss=custom_loss)

        predict = tf.keras.Model(input=[input],output=[probs])

        return policy,predict

    def choose_action(self,observation):
        state = observation[np.newaxis,:] # 2D 
        probabilities = self.predict.predict(state)[0] #model.predict(x)
        action = np.random.choice(self.action_space, p = probabilities) # action_space 까지 숫자중, p개 선택

        return action

    def store_transition(self,observation,action,reward):
        self.action_memory.append(action)
        self.state_memory.append(observation)
        self.reward_memory.append(reward)

    def learn(self):
        state_memory = np.array(self.state_memory)
        action_memory = np.array(self.action_memory)
        reward_memory = np.array(self.reward_memory)

        actions = np.zeros([len(action_memory),self.n_actions])
        actions[np.arange(len(action_memory),action_memory)] = 1

        G = np.zeros_like(reward_memory)
        for t in range(len(reward_memory)):
            G_sum = 0
            discount = 1
            for k in range(t, len(reward_memory)):
                G_sum += reward_memory[k] * discount
                discount *= self.gamma
            
            G[t] = G_sum
        mean = np.mean(G)
        std = np.std(G) if np.std(G) > 0 else 1
        self.G = (G-mean)/std

        cost = self.policy.train_on_batch([state_memory,self.G],actions)

        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []

        # return cost
    
    def save_model(self):
        self.policy.save(self.model_file)

    def load_model(self):
        self.policy = load_model(self.model_file)

import gym
import matplotlib as plt
#from utils import plotLearning
agent = Agent(ALPHA=0.0005,input_dims=8,GAMMA=0.99,n_actions=4,
                layer1_size=64,layer2_size=64)
env = gym.make('LunarLander-v2')
score_history = []

n_episodes = 2000

for i in range(n_episodes):
    done = False
    score = 0
    observation = env.reset() # 새로운 에피소드를 불러온다.

    while not done:
        action = agent.choose_action(observation)
        # 행동(action)을 취한 이후에 환경에 대해 얻은 관찰값(observation)
        # 적용하여 제어
        observation_, reward, done, info = env.step(action)
        #observation(4D) -> 
        # reward : 보상
        # done : terminal or True 인경우 
        # info : 환경의 정보들 (점수 등)

        agent.store_transition(observation,action,reward)
        observation = observation_
        score += reward
    score_history.append(score)

    agent.learn()

    print('episode ', i,'score %.1f' % score,
        'average_score %.1f' % np.mean(score_history[-100:]))

filename = 'lunar_lander.png'



