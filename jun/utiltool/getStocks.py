import yfinance as yf
from pandas_datareader import data as pdr
import math

def getStocks():
    yf.pdr_override()

    # ======== 데이터셋 구성 ======== #
    STOCK_CODE = "^KS11"
    start_date = "2020-02-01"
    end_date = "2021-04-30"
    T = 31

    STOCK_DATA = pdr.get_data_yahoo(STOCK_CODE, start=start_date, end=end_date)["Close"]

    stockArray = STOCK_DATA.to_numpy()
    
    M = math.floor(len(stockArray) / (T))
    S = stockArray[:M * (T)].reshape(M, T)
    return S

def getTotalStocks():
    # ======== 데이터셋 구성 ======== #
    STOCK_CODE = "^KS11"
    start_date = "2020-02-01"
    end_date = "2021-04-30"
    T = 31

    STOCK_DATA = pdr.get_data_yahoo(STOCK_CODE, start=start_date, end=end_date)["Close"]
    stockArray = STOCK_DATA.to_numpy()

    return stockArray.reshape(-1)
