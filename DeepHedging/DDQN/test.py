import numpy as np
from matplotlib import pyplot as plt

from DDQN.ddqn import DQNAgent
from DDQN.env_trade import TradeEnv
from DDQN.utils.getStocks import getTotalStocks

TEST_EPISODES = 100

S = getTotalStocks()
env = TradeEnv(S)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

agent = DQNAgent(state_size, action_size)
agent.load("./version/hedge-ddqn_1.1.h5")
agent.epsilon = 0.0
done = False
batch_size = 32
hist = []

for e in range(TEST_EPISODES):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    for time in range(len(S)):
        action = agent.predict(state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        state = next_state
        if done:
            print("episode: {}/{}, score: {}, e: {:.2}"
                  .format(e, TEST_EPISODES, reward, agent.epsilon))
            hist.append(reward)
            break


plt.plot(np.arange(0, len(hist)), hist, marker=".", linestyle='none')
plt.show()