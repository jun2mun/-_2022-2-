from utils._api import pandas_datareader_api_v1
from utils._math import d1,BS_Call,BS_PUT
from model.BS_test import BS_test
from model.BS_model import BlackScholes_model
from model.BS_reinforce_model import Agent, Old_Agent, BS_reinforce_model
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from utils._math import monte_carlo_paths

def init():
    STOCK_CODE = 'TSLA'
    STOCK = pandas_datareader_api_v1(STOCK_CODE,start='2012-01-01',end='2013-12-31')
    print(STOCK.head())
    STOCK_2 = pandas_datareader_api_v1(STOCK_CODE,start='2014-01-01',end='2015-12-31')
    full =  pd.concat([STOCK,STOCK_2],axis =0)

    STOCK_3 = pandas_datareader_api_v1(STOCK_CODE,start='2016-01-01',end='2017-12-31')
    full = pd.concat([full,STOCK_3],axis=0)
    STOCK_4 = pandas_datareader_api_v1(STOCK_CODE,start='2018-01-01',end='2019-12-31')
    full = pd.concat([full,STOCK_4],axis=0)
    print(full.shape) #(2014,)
    #STOCK_5 = pandas_datareader_api_v1(STOCK_CODE,start='2020-01-01',end='2021-12-31')
    
    full = full[['Close']]
    full.reset_index(drop=True,inplace=True) # (439,1)
    ''' 15일 + 결과값 * 100개 시뮬레이션 (100,16)'''
    ''' train 80% '''
    full = full[:2000]

    # 칼럼 명 변경
    full = full.rename(columns = {'Close' : 'Price_1'})

    '''
    y_train = []
    for arr in train_set:
        out = arr[-1]
        np.delete(arr,-1)
        y_train.append(out)
    y_train = np.array(y_train)
    y_train = np.reshape(y_train,(80,1))
    X_train = train_set
    '''
    test_train_size = 1600
    full_train = full[:test_train_size]
    full_test = full[test_train_size:]
    # 데이터 전처리
    window_size = 15
    for i in range(2, window_size+1) :
        full_train[f'Price_{i}'] = full_train['Price_1'].shift(i)
        full_test[f'Price_{i}'] = full_test['Price_1'].shift(i)
    #print(LCID_train.shape) # (280,16)

    
    full_train.dropna(inplace=True) # NAN 이 있는 열 날림(초기 14일치)
    #print(y_train.shape) # (266,)

    full_test.dropna(inplace=True) # NAN 이 있는 열 날림(초기 14일치)
    #print(X_train,y_train,X_train.shape,y_train.shape)
    X_train = full_train
    X_test = full_test
    
    return X_train,X_test

if __name__ == '__main__':
    # 데이터 수집 -> Train,test set 분리
    #X_train,X_test = init()

    # 입력 데이터 #
    '''
    초기 주가 : S0
    K : Strike 주가
    vol : 0.2 변동성 (임의?)
    T = 1/12 만기 까지 남은 날짜
    timesteps : shift 크기 (일자)
    seed = 42
    n_sims = 50000
    '''

    S0=100;K=100;r=0;vol=0.2;T=1/12;timestamp=30
    seed = 42; n_sims= 50000

    paths_train = monte_carlo_paths(S0,T,vol,r,seed,n_sims,timestamp)
    # paths_train.shape => (31,50000,1)
    # (1,50000) -> 1일차 데이터 50000만개

    #print(paths_train[-1,:,].shape) # (50000,1)
    #print(paths_train[-1,:,0].shape) # (50000,)
    #print(paths_train[-1,:,1]) X
    
    batch_size = 1000 
    features = 1 
    K = 100
    alpha = 0.50 #risk aversion parameter for cVaR
    epoch = 100 #It is set to 100, but should ideally be a high number 
    #hedge,alpha,cost,K,T,r,sig,time_steps,batch_size=1000
    model_1 = BS_reinforce_model(scenario= paths_train,hedge=0,alpha=alpha,cost=0,K=K,T=T,r=r,sig=0.0,time_steps=timestamp,)