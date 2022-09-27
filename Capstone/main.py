from tokenize import _all_string_prefixes
from utils._math import europian_option , BS_CALL, BS_PUT
from utils._api import pandas_datareader_api_v2
import numpy as np
def first():

    '''
    블랙숄즈 모형 (유러피안 옵션)
    europtian_option(100,100,1,0.02,0.2,'call') 기초자산의 현재가격 100, 행사가격 100, 만기를 1년, 무위험이자율을 2% , 기초자산의 변성 20%
    => 이론가 8.916
    K는 행사가
    r은 무위험이자율
    sigma 기초자산의 연간 변동성
    T 만기 (3개월, 1년 등 만기까지의 기간)
    S 기초자산의 가격
    '''
    # Parameter
    K = 100
    r = 0.01
    sigma = 0.25
    T = np.linspace(0,1,100) # 균일한 간격으로 채움
    S = np.linspace(0,200,100)
    T, S = np.meshgrid(T,S)
    print(T)



    #europian_option()


def test1():
    import math
    S0 = 100 # 현재 주가
    V = 0.3 # 주가의 기간별 변동성
    T = 5 # 이항 분포가 
    dt = 1 # 이항분포가 1회 발생하는 단위기간을 의미
    Rf = 0.05 # 무위험이자율 (이산 복리)
    K = 100 # 옵션의 행사 가격

    u = math.exp(V * math.sqrt(dt)) # 주가 상승 배수
    d = 1 / u # 주가 하락 배수
    P = (math.exp(Rf * dt) - d) / (u - d) # 위험 중립 확률
    print('u = ',u)
    print('d = ',d)
    print('P = ',P)


def test2():
    import pandas as pd
    import pandas_datareader.data as web
    import numpy as npp
    import matplotlib.pyplot as plt
    plt.style.use('ggplt')


def test3():
    import matplotlib.pyplot as plt
    K = 100
    r = 0.1
    T = 1
    sigma = 0.3
    S = np.arange(60,140,0.1)
    calls = [BS_CALL(s,K,T,r,sigma) for s in S]
    puts = [BS_PUT(s,K,T,r,sigma) for s in S]
    plt.plot(calls, label='Call Value')
    plt.plot(puts,label='Put Value')
    plt.xlabel('$S_0$')
    plt.ylabel(' Value')
    plt.legend()
    plt.show()

def test4():
    import matplotlib.pyplot as plt
    K = 100
    r = 0.1
    T = 1
    Sigmas = np.arange(0.1, 1.5, 0.01)
    S = 100

    calls = [BS_CALL(S, K, T, r, sig) for sig in Sigmas]
    puts = [BS_PUT(S, K, T, r, sig) for sig in Sigmas]
    plt.plot(Sigmas, calls, label='Call Value')
    plt.plot(Sigmas, puts, label='Put Value')
    plt.xlabel('$\sigma$')
    plt.ylabel(' Value')
    plt.legend()
    plt.show()

def test5():
    import matplotlib.pyplot as plt
    K = 100
    r = 0.05
    T = np.arange(0, 2, 0.01)
    sigma = 0.3
    S = 100

    calls = [BS_CALL(S, K, t, r, sigma) for t in T]
    puts = [BS_PUT(S, K, t, r, sigma) for t in T]
    plt.plot(T, calls, label='Call Value')
    plt.plot(T, puts, label='Put Value')
    plt.xlabel('$T$ in years')
    plt.ylabel(' Value')
    plt.legend()
    plt.show()

def test6():
    import pandas_datareader.data as web
    import pandas as pd
    import datetime as dt
    import numpy as np
    import matplotlib.pyplot as plt

    start = dt.datetime(2010,1,1)    
    end =dt.datetime(2020,10,1) 
    symbol = 'AAPL' ###using Apple as an example
    source = 'yahoo'
    data = web.DataReader(symbol, source, start, end)
    data['change'] = data['Adj Close'].pct_change()
    data['rolling_sigma'] = data['change'].rolling(20).std() * np.sqrt(255)


    data.rolling_sigma.plot()
    plt.ylabel('$\sigma$')
    plt.title('AAPL Rolling Volatility')
    plt.show()
test6()