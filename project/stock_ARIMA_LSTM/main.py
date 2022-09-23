import pandas as pd
import numpy as np
from fbprophet import Prophet
import seaborn as sns
import yfinance as yf
import FinanceDataReader as fdr
import matplotlib.pyplot as plt

def starter():
    samsung = fdr.DataReader('005930')
    print(samsung.tail())
    '''
                Open   High    Low  Close    Volume    Change
                (시가) (고가)  (저가) (종가)   (거래량)  (대비)
    Date
    2022-09-16  55600  56400  55500  56200  13456503  0.003571
    2022-09-19  56300  57000  56000  56400  12278653  0.003559
    '''

    plt.figure(figsize=(16,9))
    sns.lineplot(x=samsung.index,y='Close',data=samsung)
    plt.show()

def subploting():
    STOCK_CODE = '005930'
    stock = fdr.DataReader(STOCK_CODE)
    print(stock.head(),'\n',stock.index)

    plt.figure(figsize=(16,9))
    sns.lineplot(y=stock['Close'],x=stock.index)
    plt.xlabel('time')
    plt.ylabel('price')

    time_steps = [['1990', '2000'], 
                ['2000', '2010'], 
                ['2010', '2015'], 
                ['2015', '2020']]

    fig, axes = plt.subplots(2,2)
    fig.set_size_inches(16,9)
    for i in range(4):
        ax = axes[i//2,i%2]
        df = stock.loc[(stock.index > time_steps[i][0]) & (stock.index < time_steps[i][1])]
        sns.lineplot(y=df['Close'],x=df.index, ax = ax)
        ax.set_title(f'{time_steps[i][0]}~{time_steps[i][1]}')
        ax.set_xlabel('time')
        ax.set_ylabel('price')
    plt.tight_layout()
    plt.show()

def predicting():
    from fbprophet import Prophet
    from fbprophet.plot import plot_plotly, plot_components_plotly
    STOCK_CODE = '005930'
    stock = fdr.DataReader(STOCK_CODE, '2019')
    stock['y'] = stock['Close'] # 예측값
    stock['ds'] = stock.index # 시계열 데이터

    m = Prophet()
    m.fit(stock)

    future = m.make_future_dataframe(periods=30)
    #print(future.tail())

    forecast = m.predict(future)
    print(forecast.tail())
    print(forecast[['ds','yhat','yhat_lower','yhat_upper']].iloc[-40:20])
    plt.plot(forecast)
    plt.show()

##################################################################################

def preprocessing_ARIMA():
    '''
    samsung_df = yf.download('005930.KS',
                        start='2020-01-01',
                        end='2021-04-21',
                        progress=False)

    samsung_df = samsung_df[["Close"]]
    #print(samsung_df)
    samsung_df = samsung_df.reset_index(inplace=True) # Date index를 column로 만듬
    #print(samsung_df)
    samsung = samsung.rename(columns = { 'Close'})

    samsung_df.columns = ['day','price']
    samsung_df['day'] = pd.to_datetime(samsung_df['day'])

    samsung_df.index = samsung_df['day']
    samsung_df.set_index('day',inplace=True)
    #print(samsung_df)
    '''
    #삼성전자 주가 다운로드(2020-01-01 ~ 2021-12-01)
    samsung_df = yf.download('005930.KS', #삼성전자주 코드
                start='2020-01-01', end='2021-12-01')

    # [close == 종가]
    samsung_df = samsung_df[['Close']]
    samsung_df.reset_index(inplace=True)
    samsung_df = samsung_df.rename(columns = {'Close' : 'price'})
    samsung_df.head(3)


    samsung_train_df = samsung_df[:317]
    samsung_test_df = samsung_df[317:]

    from statsmodels.tsa.arima.model import ARIMA
    import statsmodels.api as sm

    # (AR = 2, 차분 = 1, MA = 2)
    model = ARIMA(samsung_train_df.price.values,order=(2,1,2)) 
    model_fit = model.fit()
    #model_fit = model.fit(full_output =True, disp = True) # trend = 'c'
    #print(model_fit.summary())
    #print(model_fit.predict())
    
    pred = pd.DataFrame(model_fit.predict())
    #print(pred.head())
    #print(samsung_df.head())
    plt.plot(pred[1:])
    plt.plot(samsung_df['price'])
    plt.show()

#preprocessing_ARIMA()

def preprocessing_LSTM():
    from sklearn.preprocessing import RobustScaler
    rb = RobustScaler()
    #삼성전자 주가 다운로드(2020-01-01 ~ 2021-12-01)
    samsung_df = yf.download('005930.KS', #삼성전자주 코드
                start='2020-01-01', end='2021-12-01')

    # [close == 종가]
    samsung_df = samsung_df[['Close']]
    samsung_df.reset_index(drop=True,inplace=True) # time_stamp 날리기
    samsung_df = samsung_df.rename(columns = {'Close' : 'price'})
    samsung_df.head(3)

    samsung_scaled = rb.fit_transform(samsung_df[['price']])
    samsung_df['price'] = samsung_scaled

    # train, test set 분리
    test_size = 100 # data split size
    train_data = samsung_df[:-test_size]
    test_data = samsung_df[-test_size:]

    # 당일 데이터 예측에 +n일의 과거 데이터를 반영한다.
    window_size = 15 # 예측에 반영할 과거 데이터 일수
    for i in range(1, 15) :
        train_data[f'price_{i}'] = train_data['price'].shift(i)
        test_data[f'price_{i}'] = test_data['price'].shift(i)
        # train, test 데이터를 하루 씩 옮기면서 과거 데이터를 형성
    
    # 데이터 확인
    print(train_data.shape)
    print(train_data)
    # 과거 데이터가 채워지지 않으면 drop 함
    train_data.dropna(inplace=True)
    X_train = train_data.drop('price',axis=1)
    print(X_train.shape)
    y_train = train_data[['price']]
    test_data.dropna(inplace=True)
    #print(test_data.head())
    X_test = test_data.drop('price',axis=1) # 'price' column 날리기
    #print(X_test.head())
    y_test = test_data[['price']]

    # train,test 사이즈를 확인하고, 신경망 학습을 위해 reshape 한다.
    X_train = X_train.values
    X_test = X_test.values
    y_train = y_train.values
    y_test = y_test.values
    # (360, 14) (360, 1) (86, 14) (86, 1) #
    print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)


    X_train = X_train.reshape(X_train.shape[0],14,1)
    X_test = X_test.reshape(X_test.shape[0],14,1)

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

    # 모델 학습
    es = EarlyStopping(monitor='loss', patience=5, verbose= 1)
    model.fit(X_train, y_train, epochs = 50, batch_size=16, verbose=1,callbacks=[es])

    # 모델 예측
    y_pred = model.predict(X_test)

    #model.save('LSTM_name.h5')
    from tensorflow.python.keras.models import load_model
    model = load_model("model_name.h5")
    # *-- 결과 시각화 --*
    # 예측 결과와 실제 값을 시각화
    y_test_val = pd.DataFrame(y_test, index=test_data.index)
    y_pred_val = pd.DataFrame(y_pred, index=test_data.index)

    import matplotlib.pyplot as plt
    ax1 = y_test_val.plot()
    y_pred_val.plot(ax=ax1)
    plt.legend(['test','pred'])
    plt.show()

preprocessing_LSTM()