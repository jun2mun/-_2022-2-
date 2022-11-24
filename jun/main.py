#%%
from utiltool.getStocks import getStocks, getTotalStocks

# ======== 데이터셋 구성 ======== #
#S = getTotalStocks()
S = getStocks()
5
from A3C import A3C
A3C.main(S[0])