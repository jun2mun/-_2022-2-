
def yahoo_api():
    import yfinance as yf

    pass

def FinanceData_api():
    import FinanceDataReader as fdr
    
    STOCK_CODE = '005930'
    stock = fdr.DataReader(STOCK_CODE)
    pass