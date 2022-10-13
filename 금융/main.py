from utils._api import pandas_datareader_api_v1
from utils._math import d1,BS_Call,BS_PUT
from model.BS_model import BlackScholes_model
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

if __name__ == '__main__':

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
    print(X_train.head())
    '''------------------------ 모델 생성----------------------------'''

    # 파라미터 임의로 지정
    hedge = 0; cost = 0; K = 20; T = 30/365 ; r= 0.0; sig = 0.2
    model = BlackScholes_model(hedge = 0, cost = 0, K = 23, T = 30/365, r= 0.0, sig = 0.2)



    '''----------------차원 전처리 -> train set----------------''' 
    p_list = []
    y_list = []
    SS_list = []
    ## SS 변화하는 주식 가격
    for arr in X_train['Price_1']:
        p = BS_Call(arr,K,T,r,sig)
        p_list.append(p)
    c = np.zeros([X_train.shape[0],1])
    p = np.array(p_list).reshape(X_train.shape[0],1)
    print(p.shape)
    x_train_list = [p] + [c]
    for i in range(1,16):
        temp_list = []
        for index,value in X_train['Price_'+str(i)].iteritems():
            #print(index,value)
            temp_list.append(value)
        temp_list = np.array(temp_list).reshape(-1,1)
        x_train_list.append(temp_list)   
    y = np.zeros([X_train.shape[0],1])

    '''---------------------모델 학습 & 예측 & 로드--------------------------'''

    model.compile(loss='mse',optimizer='adam')
    hist = model.fit(x_train_list,y, batch_size=32, epochs=50,  verbose=True, validation_split=0.2)
    model.save('LCID_name.h5')
    #model = tf.keras.models.load_model("LCID_name.h5")

    '''----------------차원 전처리 -> train set----------------''' 
    p_list = []
    y_list = []
    SS_list = []
    ## SS 변화하는 주식 가격
    for arr in X_test['Price_1']:
        p = BS_Call(arr,K,T,r,sig)
        p_list.append(p)
    c = np.zeros([X_test.shape[0],1])
    p = np.array(p_list).reshape(X_test.shape[0],1)
    print(p.shape)
    X_test_list = [p] + [c]
    for i in range(1,16):
        temp_list = []
        for index,value in X_test['Price_'+str(i)].iteritems():
            #print(index,value)
            temp_list.append(value)
        temp_list = np.array(temp_list).reshape(-1,1)
        X_test_list.append(temp_list)   

    y_pred = model.predict(X_test_list)
    

    '''---------------------데이터 시각화--------------------------'''
    plt.hist(y_pred,bins=30)
    plt.show()
    
    plt.plot(hist.history['loss'])
    plt.ylabel('loss')
    plt.xlabel('iter')
    plt.show()

    plt.plot(y_pred-p,marker=".",linestyle='none')
    #plt.plot(p,marker=".",linestyle='none')
    plt.xlabel('strike stock price')
    plt.ylabel('cost')
    plt.show()