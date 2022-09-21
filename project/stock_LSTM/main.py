'''
c
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
import matplotlib.pyplot as plt
from utils.training.common import ts_train_test
from model.LSTM_model import LSTM_model

AMZN = yf.download('AMZN', 
                  start = '2013-01-01',
                  end = '2019-12-31',
                  progress = False)
all_data = AMZN[['Adj Close', 'Open', 'High','Low',"Close","Volume"]].round(2)
all_data.head(10)

print("There are "+str(all_data[:'2018'].shape[0])+" observations in the training data")
print("There are "+str(all_data['2019':].shape[0])+" observations in the test data")

#plt.plot(all_data['Adj Close'])
#plt.show()

X_train, y_train, X_test = ts_train_test(all_data,5,2)
X_train.shape[0], X_train.shape[1]

# Convert the 3D shape of X_train to a data frame so we can see: 
X_train_see = pd.DataFrame(np.reshape(X_train, (X_train.shape[0], X_train.shape[1])))
y_train_see = pd.DataFrame(y_train)
pd.concat([X_train_see, y_train_see], axis = 1)

# Convert the 3D shape of X_test to a data frame so we can see: 
X_test_see = pd.DataFrame(np.reshape(X_test, (X_test.shape[0], X_test.shape[1])))
pd.DataFrame(X_test_see)

print("There are " + str(X_train.shape[0]) + " samples in the training data")
print("There are " + str(X_test.shape[0]) + " samples in the test data")


from utils.training.common import ts_train_test_normalize
from model.Simple_rnn_model import simple_rnn_model
from utils.plot._show_plot import actual_pred_plot

X_train, y_train, X_test, sc = ts_train_test_normalize(all_data, 5,2)
my_rnn_model, rnn_predictions_2 = simple_rnn_model(X_train, y_train, X_test, sc)
rnn_predictions_2[1:10]
actual_pred_plot(rnn_predictions_2,all_data)

my_LSTM_model, LSTM_prediction = LSTM_model(X_train, y_train, X_test, sc)
LSTM_prediction[1:10]
actual_pred_plot(LSTM_prediction,all_data)