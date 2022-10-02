from _api import pandas_datareader_api_v1
from _math import d1,BS_Call,BS_PUT
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

if __name__ == '__main__':

    STOCK_CODE = 'TLSA'
    LCID = pandas_datareader_api_v1(STOCK_CODE,start='2012-01-01',end='2022-01-01')
    print(LCID.shape)
    LCID = LCID[['Close']][:660].to_numpy()
    LCID = np.reshape(LCID,(-1,11))

    m = 10
    K_1 = 1.17
    K_2 = 1.67
    K_3 = 2.17

    r = 0.1; sigma=0.2; T = 10/365; dt = 1/365
    N = 10
    a = []
    print(LCID[0,0])
    for i in range(60):
        hedge = 0 # 0으로 헷지하고 싶다.
        cost = 0
        for j in range(N):
            #print(LCID[0][j])
            delta_1 = stats.norm.cdf(d1(LCID[i][j],K_1,sigma,T-j*dt,r))
            delta_2 = stats.norm.cdf(d1(LCID[i][j],K_2,sigma,T-j*dt,r))
            delta_3 = stats.norm.cdf(d1(LCID[i][j],K_3,sigma,T-j*dt,r))
            delta = delta_1 - 2* delta_2 + delta_3

            cost += (hedge-delta) * LCID[i][j] # 돈이 cost 만큼 사용된다. (-면 지출 +면 수입)
            print(LCID[0][j],delta,delta-hedge,cost) # 주가, delta(주식을 delta개 갖고 있어라) , 헷지하려면 "이만큼 사라"
            # 맨마지막 cost가 옵션의 가격
            hedge = delta
        
        cost += hedge * LCID[i][N]
        a.append(cost)
    '''
    if LCID[0][N] > K:
        #delta = 1
        cost += (hedge-1) * LCID[0][N] + K
    
    else :
        #delta = 0
        cost += (hedge-0) * LCID[0][N]
    print(f'만기시점의 주가 : {LCID[0][N]},0,{cost}')
    '''
    #print(a)
    #print(LCID[:,N])
    #plt.plot(LCID[:,-1],a,marker=".",linestyle='none')
    #plt.show()

    #print(LCID)
    #print(f'유럽형 콜옵션의 블랙숄즈 이론 가격은 : {BS_Call(LCID[0][0],K,sigma,T,r)}')
    #print(f'유럽형 풋옵션의 블랙숄즈 이론 가격은 : {BS_PUT(LCID[0][0],K,sigma,T,r)}')