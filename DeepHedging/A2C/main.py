import time
from A2C.agent import A2CAgent
from A2C.environment import TradeEnv
import numpy as np

state_size = 100
state_seq_length = 100
action_size = 100

def run_experiment():
    start = time.time()
    env = TradeEnv()
    agent = A2CAgent(state_size, state_seq_length, action_size)
    epochs = 3
    reward_hist = []

    print('Setup: {:.4f}'.format(time.time() - start))

    for e in range(epochs):

        start = time.time()
        state = env.reset()
        state = np.reshape(state, [1,state_seq_length, state_size])
        done = False
        total_reward = 0
        print('Game Start: {:.4f}'.format(time.time() - start))

        while not done:

            start = time.time()
            action = agent.get_action(state)
            print('Get Action: {:.4f}'.format(time.time() - start))

            start = time.time()
            next_state, reward, done, info = env.step(action)
            print('Step: {:.4f}'.format(time.time() - start))

            start = time.time()
            next_state = np.reshape(next_state, [1,state_seq_length, state_size])
            agent.train_model(state, action, reward, next_state, done)
            print('Train: {:.4f}'.format(time.time() - start))

            total_reward += reward
            state = next_state

        print(total_reward)
        reward_hist.append(total_reward)
    return reward_hist

# Running training takes very long

import matplotlib.pyplot as plt
reward_hist = run_experiment()
plt.plot(reward_hist)