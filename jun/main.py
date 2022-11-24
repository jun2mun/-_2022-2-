#%%
from utiltool.getStocks import getStocks, getTotalStocks
from env.env_trade import TradeEnv

# ======== 데이터셋 구성 ======== #
#S = getTotalStocks()
S = getStocks()

from A3C import A3C
env = TradeEnv(S[0])
model = A3C.Agent(env)

model.train()
