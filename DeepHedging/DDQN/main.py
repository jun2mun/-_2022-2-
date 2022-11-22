from env_trade import TradeEnv
from ddqn import DQNAgent
from utils.getStocks import getStocks,getTotalStocks
import numpy as np
import matplotlib.pyplot as plt

EPISODES = 1000
S = getTotalStocks()

env = TradeEnv(S)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

agent = DQNAgent(state_size, action_size)
# agent.load("./save/cartpole-ddqn.h5")
done = False
batch_size = 70
reward_history = []
for e in range(EPISODES):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    for time in range(len(S)):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        # reward = reward if not done else -10
        # x, x_dot, theta, theta_dot = next_state
        # r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        # r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        # reward = r1 + r2

        next_state = np.reshape(next_state, [1, state_size])
        agent.memorize(state, action, reward, next_state, done)
        state = next_state
        if done:
            agent.update_target_model()
            print("episode: {}/{}, score: {}, e: {:.2}"
                  .format(e, EPISODES, time, agent.epsilon))
            reward_history.append(time)
            break
        # if len(agent.memory) > batch_size:
        # agent.replay(batch_size)
    if (e+1) % 5 == 0:
        agent.replay(batch_size)

    if (e+1) % 20 == 0 and e != 0:
        plt.style.use('seaborn')
        plt.plot(np.arange(0, len(reward_history), 10), reward_history[::10])
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.show()
        agent.save("./save/cartpole{}-ddqn.h5".format(e+1))