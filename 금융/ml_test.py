import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Reshape, Flatten
from keras.optimizers import Adam
from keras.layers.convolutional import Convolution2D
import tensorflow as tf

class PGAgent:
    def __init__(self,state_size,action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99
        self.learning_rate = 0.001
        self.states = []
        self.gradients = []
        self.rewards = []
        self.probs = []
        self.model = self._build_model()
        self.model.summary()
    
    def _build_model(self):
        input = tf.keras.layers.Input(shape=(30,1)) # (time_steps,feature)
        # shape=(self.input_dims)
        output,state_h,state_c = tf.keras.layers.LSTM(64,return_state=True)(input)
        output = tf.keras.layers.Dense(32,activation='tanh')(output)
        # self.fc1_dims
        # self.fc2.dims
        output = tf.keras.layers.Dense(1)(output)
        # probs = Dense(self.n_actions, activation = 'softmax)(위)

        
        strategy = tf.reshape(output,(30,n_sims))
        strategy = tf.cast(strategy,tf.float64)
        option = tf.maximum(STOCK_T - STOCK_K,0)
        Hedging_PnL = - option + tf.reduce_sum(STOCK_dS * strategy, axis=0)
        Hedging_PnL_paths = - option + STOCK_dS * strategy #(10000,) + (30,10000)
        CVaR, idx = tf.math.top_k(-Hedging_PnL,k=tf.cast((1-alpha)* n_sims, tf.int32)) # must set to int32 !!
        CVaR = tf.reduce_mean(-Hedging_PnL)
        model = tf.keras.Model(inputs=input,outputs=CVaR)
        model.compile(optimizer=Adam(lr=self.lr),loss='categorical_crossentropy')
        
        return model

    # 강화학습 데이터 기억 메모리
    def memorize(self,,state,action,prob,reward):
        y = np.zeros([self.action_size]) # action size란?
        y[action] = 1
        self.gradients.append(np.array(y).astype('float32') - prob)
        self.states.append(state)
        self.rewards.append(reward)
    
    # 모르겠음
    def discount_rewards(self,rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in range(reversed(0,rewards.size)):
            if rewards[t] ! = 0:
                running_add = 0
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    def train(self):
        gradients = np.vstack(self.gradients)
        rewards = np.vstack(self.rewards)
        rewards = self.discount_rewards(rewards)
        reward = (reward - np.mean(rewards)) / (np.std(rewards) + 1e-7)
        gradients *= rewards
        X = np.squeeze(np.vstack([self.states]))
        Y = self.probs + self.learning_rate * np.squeeze(np.vstack([gradients]))
        self.model.train_on_batch(X,Y)
        self.states, self.probs, self.gradients, self.rewards = [],[],[],[]
    
    def load(self,name):
        self.model.load_model(name)
    
    def save(self,name):
        self.model.save(name)

def preprocess():
    pass


env = gym.make("Pong-v0")
state = env.reset()
prev_x = None
score = 0
episode = 0

state_size = 80 * 80
action_size = env.action_space.n
agent =PGAgent(state_size,action_size)
#agent.load('')
while True:
    env.render()

    cur_x = preprocess(state)
    x = cur_x - prev_x if prev_x is not None else np.zeros(state_size)
    prev_x = cur_x

    action, prob = agent.act(x)
    state, reward, done, info = env.step(action)
    score += reward
    agent.memorize(x, action, prob, reward)

    if done:
        episode += 1
        agent.train()
        print('Episode: %d - Score: %f.' % (episode, score))
        score = 0
        state = env.reset()
        prev_x = None
        if episode > 1 and episode % 10 == 0:
            agent.save('pong.h5')