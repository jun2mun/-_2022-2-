from utils._api import pandas_datareader_api_v1
from utils._math import d1,BS_Call,BS_PUT,monte_carlo_paths
from model.BS_model import BlackScholes_model
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import tensorflow as tf

def _init():
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

    print(full)



def arc_init():
    S_0 = 100
    K = 100
    r = 0
    vol = 0.2
    T = 1/12
    timesteps = 30
    seed = 42
    n_sims = 50000

    paths_train = monte_carlo_paths(S_0,T,vol,r,seed,n_sims,timesteps)
    return paths_train


def preproc():
    STOCK_GRAPH = arc_init()
    #print(STOCK_GRAPH.shape) # (31,50000,1) 31일간 시나리오를 50000개 갖고 있다.

    STOCK_T = STOCK_GRAPH[-1,:,0]
    STOCK_dS = STOCK_GRAPH[1:,:,0] - STOCK_GRAPH[0:-1,:,0]
    STOCK_K = np.full((50000,1),100)
    STOCK_GRAPH = tf.unstack(STOCK_GRAPH[:-1,:,:],axis=0)
    STOCK_GRAPH = tf.stack(STOCK_GRAPH)
    STOCK_GRAPH = np.swapaxes(STOCK_GRAPH,0,1)
    return STOCK_GRAPH,STOCK_K,STOCK_T
    #print("안녕하세요",type(STOCK_GRAPH))
    #inputs = tf.random.normal([30,50000, 1])
    #STOCK_GRAPH = tf.unstack(STOCK_GRAPH[:-1,:,:],axis=0)

    #print(STOCK_GRAPH)    

    '''
    # strategy 결과값 -> list (batch_size,1) * 30개  -=-> strategy 값이 delta 같은 느낌.
    # state 결과값 -> (batch_size,n[0]), (batch_size,n[1]), (batch_size,n[2]), (batch_size,n[3])
    strategy, state = tf.compat.v1.nn.static_rnn(lstm,STOCK_GRAPH,initial_state=lstm.zero_state(50000,tf.float64),dtype=tf.float64)


    strategy = tf.reshape(strategy,(30,50000))
    #print(strategy.shape)
    option = tf.maximum(STOCK_T- STOCK_K,0)

    #print(option)
    Hedging_PnL = option + tf.reduce_sum(STOCK_dS * strategy,axis=0)
    #print(Hedging_PnL)
    Hedging_PnL_Paths = - option + STOCK_dS * strategy
    #print(Hedging_PnL_Paths)

    alpha = 0.50 # risk aversion parameter for CVaR
    # top k value, indices of these values in the origin tensor
    CVaR , idx = tf.compat.v1.nn.top_k(-Hedging_PnL,tf.cast((1-alpha)*50000,tf.int32))
    #print(CVaR,idx) # CVaR  : (30,50000) | idx : (25000,)
    CVaR = tf.reduce_mean(CVaR)
    print(CVaR) # tf.Tensor(-0.26285629929618515, shape=(), dtype=float64)
    print("stop")
    #print(type(STOCK_GRAPH),len(STOCK_GRAPH)) # LIST, 30 길이
    
    #S_T = Stock_graph[-1] # T 시점에서 주가
    #dS = Stock_graph[] # 주가 변동성  그래프로 표시.
    #S_t 는 Stock_graph 에서 S_T  를 삭제한 그래프

    #lstm = tf.keras.layers.LSTM(10)

    # lstm.zero_state = > initial_state , 
    # static_rnn

    # strategy
    # option -> 이게 의미하는 봐 마지막날, K랑 T랑 비교하였을 때 누가 더 크냐

    # Hedging_PnL = - option + dS * strategy
    # HEdging_PnL_paths =  - option + dS * strategy

    # CVaR, idx => Hedging_PnL , tf.cast((1-alpha) * batch_size)
    # CVaR = > tf.reduce_mean(CVaR)

    # 나머지는 optimizer 함수
    # 모델 저장
    '''

def deep_lstm_model(STOCK_T,STOCK_K):
    
    input = tf.keras.layers.Input(shape=(30,1)) # (time_steps,feature)
    # cell_state,hidden_state || last_time step hidden state (*2) , cell_state for last time step
    output,state_h,state_c = tf.keras.layers.LSTM(64,return_state=True,dtype='float64')(input)
    #print(output,state_h,state_c)
    output = tf.keras.layers.Dense(1)(output)
    strategy = tf.reshape(output,(30,1000))
    option = tf.maximum(STOCK_T - STOCK_K,0)
    print(option)
    model = tf.keras.Model(inputs=input,outputs=[output,state_h,state_c])
    #model.summary()
    print('stop')
    opt = tf.keras.optimizers.Adam(learning_rate=0.1)
    opt.minimize()
    return model
graph,K,result = preproc()
model = deep_lstm_model(result,K)
model.compile(loss='mse',optimizer='adam')
result = model.predict(graph)
#hist = model.fit(graph,K)

plt.plot(result,marker=".",linestyle='none')
plt.show()
print("complete")