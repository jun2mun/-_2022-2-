import yfinance as yf
from pandas_datareader import data as pdr
import math
#from agent import custom_agent
from test import custom_agent

yf.pdr_override()

# ======== 데이터셋 구성 ======== #
STOCK_CODE = "^KS11"
start_date = "2020-02-01"
end_date = "2021-04-30"
T = 31

STOCK_DATA = pdr.get_data_yahoo(STOCK_CODE, start=start_date, end=end_date)["Close"]
stockArray = STOCK_DATA.to_numpy()
M = math.floor(len(stockArray)/(T))
S = stockArray[:M*(T)].reshape(M,T)
balance = 10000

STOCK_CODE = "^KS11"
start_date = "2021-02-01"
end_date = "2022-04-30"
T2 = 31
STOCK_DATA2 = pdr.get_data_yahoo(STOCK_CODE, start=start_date, end=end_date)["Close"]
stockArray2 = STOCK_DATA.to_numpy()
M2 = math.floor(len(stockArray)/(T2))
S2 = stockArray[:M2*(T2)].reshape(M2,T2)
balance = 10000

from A3C_Continuous import main
#main(S[0],S2[0])
custom_agent(S[0],S2[0])
