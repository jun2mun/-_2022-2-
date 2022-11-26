#%%
from utiltool.getStocks import getStocks, getTotalStocks
from env.env_trade import TradeEnv

# ======== 데이터셋 구성 ======== #
#S = getTotalStocks()
S = getStocks()

from A2C import A2C
env = TradeEnv(S[0])
model = A2C.Agent(env)

model.train()
