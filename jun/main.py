#%%
from utiltool.getStocks import getStocks, getTotalStocks
from env.env_trade import TradeEnv



from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"


# ======== 데이터셋 구성 ======== #
#S = getTotalStocks()
S = getStocks()

from A2C import A2C
env = TradeEnv(S[0])
model = A2C.Agent(env)

model.train()
