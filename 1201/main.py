#%%
from utiltool.getStocks import getStocks, getTotalStocks
from environment.env import TradeEnv



from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"


# ======== 데이터셋 구성 ======== #
from utiltool.getStocks import getStockDataToTrain
S = getStocks()

from model.A2C import Agent
env = TradeEnv(S[0])
model = Agent(env)

model.train()
