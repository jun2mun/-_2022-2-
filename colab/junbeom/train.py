import shutil

import tensorflow as tf
import yfinance as yf
from pandas_datareader import data as pdr
import math
from environment import TradeEnv
import os
from main import train

yf.pdr_override()

def get_global_train_step():
    last_train_step = 1
    file_names = os.listdir("./colab/checkpoint")
    for filename in file_names:
        name = os.path.splitext(filename)[0]
        if name.find("checkpoint") != -1:
            if int(name[-1]) > last_train_step: last_train_step = int(name[-1])

    return last_train_step

# ======== 데이터셋 구성 ======== #
STOCK_CODE = "^KS11"
start_date = "2021-02-01"
end_date = "2022-04-30"
T = 31
offset = get_global_train_step()

STOCK_DATA = pdr.get_data_yahoo(STOCK_CODE, start=start_date, end=end_date)["Close"]
stockArray = STOCK_DATA.to_numpy()
M = math.floor(len(stockArray)/(T))
S = stockArray[:M*(T)].reshape(M,T)
balance = 10000

# train latest model file
train(S[0], tf.Variable(get_global_train_step()))


#
# for i in range(M):
#     global_train_step = tf.Variable(get_global_train_step())
#     train(S[global_train_step - offset], global_train_step)
#
#     if(i != M-1):
#         idx = global_train_step.numpy()
#         shutil.copytree("./colab/checkpoint/{}".format("checkpoint" + str(idx)),
#                         "./colab/checkpoint/{}".format("checkpoint" + str(idx + 1)))
