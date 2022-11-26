#%%
from utiltool.getStocks import getStocks, getTotalStocks
from env.env_trade import TradeEnv

# ======== 데이터셋 구성 ======== #
#S = getTotalStocks()
S = getStocks()

from A2C import A2C
env = TradeEnv(S[0])
env.reset()
action = env.action_space.sample()
next_state, reward, done, _ = env.step(action)
observation, reward, done, info = env.step(action)

one = 1; two = 2; three = 3
test_list = [one,two,three]
print(test_list)
one = 4; two = 3
print(test_list)

