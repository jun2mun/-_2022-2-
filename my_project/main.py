#from sklearn.preprocessing import RobustScaler
from utils.BlackScholes import BlackScholes
from model.BlackScholes_model import BlackScholes_model
from utils._api import pandas_datareader_api_v1
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

# LUCID GROUP의 실제 데이터를 바탕으로 블랙숄즈 모형 적용
if __name__ == '__main__':
    '''--------------------  데이터 전처리 ----------------------'''
    '''
                    High     Low   Open  Close    Volume  Adj Close
    Date
    2022-09-22  15.58  14.120  15.45  14.31  20375900      14.31
    2022-09-23  14.25  13.580  14.17  14.03  18603100      14.03
    2022-09-26  14.65  13.952  14.05  14.06  11486300      14.06
    2022-09-27  15.18  14.150  14.81  14.41  20173200      14.41
    2022-09-28  15.32  14.355  14.52  15.22  15050500      15.22
    '''

    STOCK_CODE = 'LCID'
    LCID = pandas_datareader_api_v1(STOCK_CODE,start='2021-01-01',end='2022-09-29')

    LCID = LCID[['Close']]
    LCID.reset_index(drop=True,inplace=True) # (439,1)

    # 칼럼 명 변경
    LCID = LCID.rename(columns = {'Close' : 'Price'})
    S0 = LCID['Price'][0]

    #from sklearn.preprocessing import RobustScaler
    #rb = RobustScaler()
    #LCID_scaled = rb.fit_transform(LCID[['Price']])
    #LCID['Price'] = LCID_scaled
    
    # 데이터 분리 (테스트, 훈련 셋 분리) 임시
    LCID_train = LCID[:280]
    LCID_test = LCID[280:]
    rest_period  = 15
    
    # 데이터 전처리
    window_size = 15
    for i in range(1, 15) :
        LCID_train[f'Price_{i}'] = LCID_train['Price'].shift(i)
        LCID_test[f'Price_{i}'] = LCID_test['Price'].shift(i)
    #print(LCID_train.shape) # (280,16)

    LCID_train.dropna(inplace=True) # NAN 이 있는 열 날림(초기 14일치)
    X_train = LCID_train.drop('Price',axis=1) # 이전 14일치 값
    y_train = LCID_train['Price'] # 실제 값
    #print(y_train.shape) # (266,)

    LCID_test.dropna(inplace=True) # NAN 이 있는 열 날림(초기 14일치)
    X_test = LCID_test.drop('Price',axis=1) # 이전 14일치 값
    y_test = LCID_test['Price'] # 실제 값
    print(y_test)
    #print(X_test.shape,y_test.shape) # (145,14) (145,)

    '''------------------------ 모델 생성----------------------------'''

    # 파라미터 임의로 지정
    hedge = 0; cost = 0; K = 23; T = 30/365 ; r= 0.0; sig = 0.2
    model = BlackScholes_model(hedge = 0, cost = 0, K = 23, T = 30/365, r= 0.0, sig = 0.2)

    '''----------------차원 전처리 -> train set----------------''' 
    p_list = []
    y_list = []
    SS_list = []
    ## SS 변화하는 주식 가격
    for index,value in y_train.iteritems():
        p = BlackScholes.bscall(value,K,T,r,sig)
        p_list.append(p)
        y_list.append( - np.maximum(value - K, 0)+ p)
    c = np.zeros([X_train.shape[0],1])
    p = np.array(p_list).reshape(266,1)    
    x_train_list = [p] + [c] 
    for i in range(1,15):
        temp_list = []
        for index,value in X_train['Price_'+str(i)].iteritems():
            #print(index,value)
            temp_list.append(value)
        temp_list = np.array(temp_list).reshape(266,1)
        x_train_list.append(temp_list)   
    y = np.array(y_list).reshape(266,1)

    '''----------------차원 전처리 -> test set----------------''' 
    p_list = []
    y_list = []
    SS_list = []
    ## SS 변화하는 주식 가격
    for index,value in y_test.iteritems():
        #print(index,value) # index , 주식 값
        #print(BlackScholes.bscall(value,K,T,r,sig)) # (266,)
        p = BlackScholes.bscall(value,K,T,r,sig)
        p_list.append(p)
        y_list.append( - np.maximum(value - K, 0)+ p)
    c = np.zeros([X_test.shape[0],1])
    p = np.array(p_list).reshape(146,1)    
    x_test_list = [p] + [c] 
    for i in range(1,15):
        temp_list = []
        for index,value in X_test['Price_'+str(i)].iteritems():
            #print(index,value)
            temp_list.append(value)
        temp_list = np.array(temp_list).reshape(146,1)
        x_test_list.append(temp_list)
    
    '''---------------------모델 학습 & 예측 & 로드--------------------------'''

    model.compile(loss='mse',optimizer='adam')
    hist = model.fit(x_train_list,y, batch_size=32, epochs=1000,  verbose=True, validation_split=0.2)
    model.save('LCID_name.h5')
    #model = tf.keras.models.load_model("LCID_name.h5")
    y_pred = model.predict(x_test_list)
    

    '''---------------------데이터 시각화--------------------------'''
    #plt.hist(y_pred,bins=30)
    #plt.show()
    
    plt.plot(hist.history['loss'])
    plt.ylabel('loss')
    plt.xlabel('iter')
    plt.show()
    print(p)
    plt.plot(y_test,y_pred,marker=".",linestyle='none')
    #plt.plot(p,marker=".",linestyle='none')
    plt.xlabel('strike stock price')
    plt.ylabel('cost')
    plt.show()