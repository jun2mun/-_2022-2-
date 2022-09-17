from utils.D_H_math import monte_carlo_paths
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np
from utils.agent import Agent
import tensorflow._api.v2.compat.v1 as tf # tf 1버전으로 진행
###
#tf.placeholder() is not compatible with eager execution.
tf.compat.v1.disable_eager_execution()
###

if __name__ == "__main__":
    # S는 주가 # r는 감마 
    S_0 = 100; K=100; r = 0; vol = 0.2; T = 1/12
    timesteps = 30; seed = 42; n_sims = 5000
    # 몬테카를로 경로 생성하기
    # paths_train (31,5000,1)
    paths_train = monte_carlo_paths(S_0, T, vol,r,seed,n_sims,timesteps)

    # 데이터 시각화 #
    #print(paths_train[1]) # [[100.]*500개]
    #print(paths_train[0].shape) # (5000,1) 
    #print(paths_train.shape) # (5000,1) 100.

    '''
    plt.figure(figsize=(20,10))
    plt.plot(paths_train[1:31,1]) # slicing 2-D Arrays
    plt.xlabel('Time Steps')
    plt.title('Stock Price Sample Paths')
    plt.show()
    '''
    
    # 정책 경사 스크립트 #
    batch_size = 1000
    features =1
    K=100
    alpha = 0.50 # CVaR에 대한 위험 회피 매개변수
    epoch = 101 # 11로 설정되었지만 이상적으로는 높은 숫자여야 함.
    model_1 = Agent(paths_train.shape[0],batch_size,features,name='rnn_final')
    # 모델 훈련에는 수분 소요
    start = dt.datetime.now()
    with tf.Session() as sess:
        # 모델 훈련
        model_1.training(paths_train,np.ones(paths_train.shape[1])*K, alpha,\
            epoch,sess)
    print('Training finished, Time elapsed:',dt.datetime.now()-start)
