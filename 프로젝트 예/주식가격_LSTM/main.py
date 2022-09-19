'''
from statsmodels.graphics.tsaplots import plot_pacf , plot_acf 
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.stattools import pacf 
from statsmodels.regression.linear_model import yule_walker 
#from statsmodels.tsa.stattools import adfuller 
import matplotlib.pyplot as plt 
import numpy as np 
#%matplotlib inline

# 데이터 생성하기 
ar = np.array([1, -0.8, 0.2])
ma = np.array([1])
my_simulation = ArmaProcess(ar,ma).generate_sample(nsample=100)

plt.figure(figsize=[10,5]) ; # 그림의 크기 설정 
plt.plot(my_simulation, linestyle = '-', marker = 'o', color = 'b')
plt.title("Simulated Process")
plt.show()
'''

import numpy as np
# !pip install yfinance
import pandas as pd 
import yfinance as yf 
AMZN = yf.download('AMZN', 
                  start = '2013-01-01',
                  end = '2019-12-31',
                  progress = False)
all_data = AMZN[['Adj Close', 'Open', 'High','Low',"Close","Volume"]].round(2)
all_data.head(10)

print("There are "+str(all_data[:'2018'].shape[0])+" observations in the training data")
print("There are "+str(all_data['2019':].shape[0])+" observations in the test data")
all_data['Adj Close'].plot()


def ts_train_test(all_data, time_steps, for_periods): 
    """
    input:
     data: dataframe with dates and price data
    output:
     X_train, y_train: data from 2013/1/1-2018-12/31 
     X_test : data from 2019 - 
    time_steps: # of the input time steps 
    for_periods: # of the output time steps 
    """
    # create training and test set 
    ts_train = all_data[:'2018'].iloc[:,0:1].values
    ts_test = all_data['2019':].iloc[:,0:1].values
    ts_train_len = len(ts_train)
    ts_test_len = len(ts_test)
    
    # create training data of s samples and t time steps 
    X_train = [] 
    y_train = [] 
    y_train_stacked = [] 
    for i in range(time_steps, ts_train_len - 1): 
        X_train.append(ts_train[i-time_steps:i,0])
        y_train.append(ts_train[i:i+for_periods,0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    
    # Reshapng X_train for efficient modelling 
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1],1))
    
    # Preparing to creat X_test 
    inputs = pd.concat((all_data["Adj Close"][:'2018'], all_data["Adj Close"]['2019':]), axis=0).values
    inputs = inputs[len(inputs)-len(ts_test) - time_steps:]
    inputs = inputs.reshape(-1,1)
    
    X_test = []
    for i in range(time_steps, ts_test_len+ time_steps- for_periods):
        X_test.append(inputs[i-time_steps:i,0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))
    
    return X_train, y_train, X_test 

X_train, y_train, X_test = ts_train_test(all_data,5,2)
X_train.shape[0], X_train.shape[1]