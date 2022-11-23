from env.env_continuous import TradeEnv 
from utils.getStocks import getStocks, getTotalStocks

# ======== 데이터셋 구성 ======== #
S = getTotalStocks()

from A3C import A3C
A3C.main(S)
