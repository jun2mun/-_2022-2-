import yfinance as yf
from pandas_datareader import data as pdr
yf.pdr_override()


def getStockDataAll(STOCK_CODE, start, end):
    return pdr.get_data_yahoo(STOCK_CODE, start, end)

def getStockDataColumn(STOCK_CODE, start, end, column):
    return pdr.get_data_yahoo(STOCK_CODE, start, end)[column]

def getStockDataToTrain(STOCK_CODE, start_date, end_date):

    STOCK_DATA_CLOSE = getStockDataColumn(STOCK_CODE, start_date, end_date, column="Close")
    stockArray = STOCK_DATA_CLOSE.to_numpy()
    return stockArray
