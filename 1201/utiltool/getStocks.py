import yfinance as yf
import pandas as pd
from ta import volatility
from ta import trend
from ta import momentum
from pandas_datareader import data as pdr
import numpy as np
from sklearn import preprocessing
import math
yf.pdr_override()

KOSPI_STOCK_CODE = "^KS11"
SP500_STOCK_CODE = "^GSPC"
WINDOW_SIZE = 30

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
def getStockDataAll(STOCK_CODE,start,end):
    return pdr.get_data_yahoo(STOCK_CODE,start,end)

def getStockDataToTrain(STOCK_CODE, start_date, end_date):
    #OHLCV data 수집
    target = getStockDataAll(STOCK_CODE, start_date, end_date)
    open = target["Open"]
    high = target["High"]
    low = target["Low"]
    close = target["Close"]
    volume = target["Volume"]

    #kospi, sp500 수집
    kospi_data = getStockDataAll(KOSPI_STOCK_CODE, start_date, end_date)["Close"].rename("Kospi")
    sp500_data = getStockDataAll(KOSPI_STOCK_CODE, start_date, end_date)["Close"].rename("Sp500")

    #rsi, bol,macd,atr 수집
    rsi = getIndicator_rsi(target)
    macd = getIndicator_macd(target)
    bollinger = getIndicator_bollinger(target)
    atr = getIndicator_atr(target)

    #data preprocessed => ['Open', 'High', 'Low', 'Close', 'Volume', 'Kospi', 'Sp500', 'Macd','Rsi', 'Bollinger', 'Atr']
    df = pd.concat([open, high, low, close, volume, kospi_data, sp500_data, macd, rsi, bollinger, atr], axis=1)[WINDOW_SIZE:]
    OHLCV = minmax_scale(df[['Open', 'High', 'Low', 'Close']], min=0, max=10)
    OPTIONAL = minmax_scale(df[['Volume', 'Kospi', 'Sp500', 'Macd','Rsi', 'Bollinger', 'Atr']], min=0, max=3)

    prices = np.reshape(close[WINDOW_SIZE:].to_numpy(), (-1, 1))
    train_set = np.append(OHLCV, OPTIONAL, axis=1)
    train_set = np.append(train_set, prices, axis=1)

    return train_set

def getIndicator_bollinger(data):
    bol_avg = volatility.bollinger_mavg(data["Close"]).rename("Bollinger")
    return bol_avg


def getIndicator_rsi(data):
    rsi = momentum.rsi(data["Close"]).rename("Rsi")
    return rsi

def getIndicator_macd(data):
    macd = trend.macd(data["Close"]).rename("Macd")
    return macd

def getIndicator_atr(data):
    atr = volatility.average_true_range(data["High"], data["Low"],  data["Close"]).rename("Atr")
    return atr

def minmax_scale(data, min=0, max=1):
    scaler = preprocessing.MinMaxScaler(feature_range=(min, max))  ## 각 칼럼 데이터 값을 [min,max] 범위로 변환
    scaler.fit(data)  ## 각 칼럼 데이터마다 변환할 함수 생성
    transformed_X = scaler.transform(data)  ## fit에서 만들어진 함수를 실제로 데이터에 적용
    return transformed_X
