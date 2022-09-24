def yahoo_api():
    import yfinance as yf
    """
    API 호출 시
                    Open     High      Low    Close     Adj Close    Volume
                    (시가) (고가)     (저가)  (종가)     (조정 종가)  (거래량)
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

def pandas_datareader_api():
    import pandas_datareader.data as web
    from datetime import datetime, date
    import numpy as np
    import pandas as pd
    """
                        High         Low        Open       Close       Volume   Adj Close
                         (고가)       (저가)        (시가)    (종가)   (거래량)  (조정 종가)
    Date
    2021-09-24  444.670013  441.209991  441.440002  443.910004   62094800.0  437.292877
    2021-09-27  444.049988  441.899994  442.809998  442.640015   61371100.0  436.041809
    2021-09-28  440.040009  432.940002  439.690002  433.720001  130436300.0  427.254761
    2021-09-29  437.040009  433.850006  435.190002  434.450012   82329200.0  427.973877
    2021-09-30  436.769989  428.779999  436.019989  429.140015  140506000.0  422.743042
    """
    stock = 'SPY'
    expiry = '12-18-2022'
    strike_price = 370

    today = datetime.now()
    one_year_ago = today.replace(year=today.year-1)

    df = web.DataReader(stock, 'yahoo', one_year_ago, today)

    print(df.head())
