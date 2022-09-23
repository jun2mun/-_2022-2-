def yahoo_api():
    import yfinance as yf
    """
    API 호출 시
                    Open     High      Low    Close     Adj Close    Volume
    Date
    2020-01-02  55500.0  56000.0  55000.0  55200.0  51244.253906  12993228
    2020-01-03  56000.0  56600.0  54900.0  55500.0  51522.757812  15422255
    2020-01-06  54900.0  55600.0  54600.0  55500.0  51522.757812  10278951
    2020-01-07  55700.0  56400.0  55600.0  55800.0  51801.257812  10009778
    2020-01-08  56200.0  57400.0  55900.0  56800.0  52729.589844  23501171
    """
    samsung_df = yf.download('005930.KS', #삼성전자주 코드
                start='2020-01-01', end='2021-12-01')
    print(samsung_df.head())

def FinanceData_api(STOCK_CODE='005930'):
    """
    API 호출 시
                Open   High    Low  Close    Volume    Change
                (시가) (고가)  (저가) (종가)   (거래량)  (대비)
    Date
    2022-09-16  55600  56400  55500  56200  13456503  0.003571
    2022-09-19  56300  57000  56000  56400  12278653  0.003559
    """
    import FinanceDataReader as fdr
    
    STOCK_CODE = '005930'
    stock = fdr.DataReader(STOCK_CODE)
    print(stock.head())

yahoo_api()