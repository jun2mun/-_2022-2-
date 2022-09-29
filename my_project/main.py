from sklearn.preprocessing import RobustScaler
from utils.BlackScholes import BlackScholes
from utils._api import yahoo_api, pandas_datareader_api_v1
from scipy import stats
import numpy as np

if False: #__name__ == '__main__':
    '''-----------  S =>  주식 가격 구하기 ------------'''
    
    # LUCID GROUP의 실제 데이터를 바탕으로 블랙숄즈 모형 적용
    STOCK_CODE = 'LCID'
    LCID = pandas_datareader_api_v1(STOCK_CODE,start='2021-01-01',end='2022-09-29')
    #print(LCID.tail())
    '''
                    High     Low   Open  Close    Volume  Adj Close
    Date
    2022-09-22  15.58  14.120  15.45  14.31  20375900      14.31
    2022-09-23  14.25  13.580  14.17  14.03  18603100      14.03
    2022-09-26  14.65  13.952  14.05  14.06  11486300      14.06
    2022-09-27  15.18  14.150  14.81  14.41  20173200      14.41
    2022-09-28  15.32  14.355  14.52  15.22  15050500      15.22
    '''
    LCID = LCID[['Close']]
    LCID.reset_index(drop=True,inplace=True)
    #print(LCID) # (439,1)

    # 칼럼 명 변경
    LCID = LCID.rename(columns = {'Close' : 'Price'})

    from sklearn.preprocessing import RobustScaler
    rb = RobustScaler()

    LCID_scaled = rb.fit_transform(LCID[['Price']])
    LCID['Price'] = LCID_scaled
    # 데이터 분리
    LCID_train = LCID[:280]
    LCID_test = LCID[280:]


    # 과거 데이터 반영 사이즈 설정 (RNN)
    window_size = 15
    for i in range(1, 15) :
        LCID_train[f'Price_{i}'] = LCID_train['Price'].shift(i)
        LCID_test[f'Price_{i}'] = LCID_test['Price'].shift(i)
    #sprint(LCID_train.tail())
    #print(LCID_train.shape) # (280,16)

    LCID_train.dropna(inplace=True) # NAN 이 있는 열 날림(초기 14일치)
    X_train = LCID_train.drop('Price',axis=1) # 이전 14일치 값
    y_train = LCID_train['Price'] # 실제 값
    #print(y_train.shape) # (266,)
    #print(X_train) # (266,15) column (Date, Price_i 1~i~14)

    LCID_test.dropna(inplace=True) # NAN 이 있는 열 날림(초기 14일치)
    X_test = LCID_test.drop('Price',axis=1) # 이전 14일치 값
    y_test = LCID_test['Price'] # 실제 값
    #print(y_test) # (145,)
    #print(X_train) # (145,15)


    # train,test 사이즈를 확인하고, 신경망 학습을 위해 ndarray형태로 변경 한다.
    X_train = X_train.values
    X_test = X_test.values
    y_train = y_train.values
    y_test = y_test.values
    # (266, 14) (266,) (145, 14) (145, ) #
    print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)


    X_train = X_train.reshape(X_train.shape[0],14,1) #size timestamp feature
    X_test = X_test.reshape(X_test.shape[0],14,1)


    '''---------------  학습 모델 -------------- '''

        # *-- 신경망 생성 --*
    # Sequential 모델 사용
    from tensorflow.python.keras.models import Sequential
    from tensorflow.python.keras.layers import Dense, LSTM
    from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
    from tensorflow.python.keras import backend as K
    K.clear_session()

    model = Sequential()
    model.add(LSTM(15,return_sequences=True,input_shape = (14,1)))
    model.add(LSTM(28,return_sequences=False))
    model.add(Dense(1,activation='linear'))

    model.compile(loss='mean_squared_error',optimizer='adam')
    model.summary()

    #es = EarlyStopping(monitor='loss', patience=5, verbose= 1)
    
    # train 모델로 학습
    model.fit(X_train, y_train, epochs = 1000, batch_size=16, verbose=1)#,callbacks=[es])
    
    # 모델 예측
    y_pred = model.predict(X_test)
    
    #model.save('LSTM_name.h5')
    from tensorflow.python.keras.models import load_model
    import pandas as pd
    model = load_model("model_name.h5")
    # *-- 결과 시각화 --*
    # 예측 결과와 실제 값을 시각화
    y_test_val = pd.DataFrame(y_test, index=LCID_test.index)
    y_pred_val = pd.DataFrame(y_pred, index=LCID_test.index)
    print(y_test_val.head())
    print('----------------')
    print(y_pred_val.head())
    
    import matplotlib.pyplot as plt
    plt.plot(y_test_val)
    plt.plot(y_pred_val)
    plt.legend(['test','pred'])
    plt.show()

if __name__ == '__main__':
    '''-----------  S =>  주식 가격 구하기 ------------'''
    
    # LUCID GROUP의 실제 데이터를 바탕으로 블랙숄즈 모형 적용
    STOCK_CODE = 'LCID'
    LCID = pandas_datareader_api_v1(STOCK_CODE,start='2021-01-01',end='2022-09-29')
    #print(LCID.tail())
    '''
                    High     Low   Open  Close    Volume  Adj Close
    Date
    2022-09-22  15.58  14.120  15.45  14.31  20375900      14.31
    2022-09-23  14.25  13.580  14.17  14.03  18603100      14.03
    2022-09-26  14.65  13.952  14.05  14.06  11486300      14.06
    2022-09-27  15.18  14.150  14.81  14.41  20173200      14.41
    2022-09-28  15.32  14.355  14.52  15.22  15050500      15.22
    '''
    LCID = LCID[['Close']]
    LCID.reset_index(drop=True,inplace=True)
    #print(LCID) # (439,1)

    # 칼럼 명 변경
    LCID = LCID.rename(columns = {'Close' : 'Price'})
    S0 = LCID['Price'][0]
    print(S0)
    rest_period  = 15
    hedge = 0; cost = 0; K = 23
    for i in range(rest_period):
        S0 = LCID['Price'][i]
        #print(S0,end="|")
        d1 = BlackScholes._d1(S0,K,30/365 - i * 1/365,0.0,0.2)
        d2 = BlackScholes._d2(S0,K,30/365 - i * 1/365,0.0,0.2)
        
        delta = stats.norm.cdf(d1)
        delta_2 = stats.norm.cdf(d2)
        cost += (hedge - delta) * S0
        # hedge - delta 만큼 주식을 사라
        hedge = delta
    #print("\n ----------")
    if S0 > 11:
        cost += (hedge -1) * S0 + K # 돈이 들어옴 (+)
    else :
        cost += (hedge -0) * S0 
    #print(S0,cost,hedge)
    # 돈이 들어왔다 나갔다 반복해서 14일후 결과 값이 나온다.


    import tensorflow as tf
    from tensorflow.python.keras.layers import Input,Dense,Add

    my_input = []

    #premium = tf.keras.layers.Input(shape=(1,), name="premium")
    
    hedge_cost = Input(shape=(1,), name='hedge_cost')
    my_input = my_input + [hedge_cost]
    price = Input(shape=(1,), name="price")
    #my_input = my_input + [premium] + [hedge_cost] + [price]
    N = 30
    for j in range(N):
        
        delta = Dense(32, activation='tanh')(price)
        hedge_cost = Add()
        
        '''
        # delta = tf.keras.layers.BatchNormalization()(delta)
        # delta = tf.keras.layers.Dense(32, activation='leaky_relu')(delta)
        # delta = tf.keras.layers.BatchNormalization()(delta)
        # delta = tf.keras.layers.Dense(32, activation='leaky_relu')(delta)
        delta = tf.keras.layers.Dense(1)(delta)

        new_price = tf.keras.layers.Input(shape=(1,), name='S'+str(j+1))
        my_input = my_input + [new_price]

        price_inc = tf.keras.layers.Subtract(name='price_inc_'+str(j))([price, new_price])
        cost = tf.keras.layers.Multiply(name="stock_"+str(j))([delta, price_inc])
        hedge_cost = tf.keras.layers.Add(name='cost_'+str(j))([hedge_cost, cost])
        #info_set = tf.keras.layers.Concatenate()([price, new_price])
        price = new_price


        payoff = tf.keras.layers.Lambda(lambda x : tf.math.maximum(x-K,0), name='payoff')(price)
        cum_cost = tf.keras.layers.Add(name="hedge_cost_plus_payoff")([hedge_cost, payoff])
        cum_cost = tf.keras.layers.Subtract(name="cum_cost-premium")([cum_cost, premium])

        model = tf.keras.Model(inputs=my_input, outputs=cum_cost)
        '''
