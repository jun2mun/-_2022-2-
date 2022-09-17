#import tensorflow as tf
import tensorflow._api.v2.compat.v1 as tf # tf 1버전으로 진행
import numpy as np
import datetime as dt
#from IPython.core.debugger import set_trace

class Agent(object):
    ### cVAR : 손실 금액이 VaR를 넘어섰을 때 기대되는 평균적인 손실 규모
    def __init__(self,time_steps,batch_size, features,\
        nodes = [62,46,46,1], name='model'):
        # 1. 변수 초기화 ??
        tf.reset_default_graph()
        self.batch_size = batch_size # 배치의 옵션 수
        self.S_t_input = tf.placeholder(tf.float32,[time_steps,batch_size,\
            features]) # 스팟
        self.K = tf.placeholder(tf.float32,batch_size) # 행사가
        self.alpha = tf.placeholder(tf.float32) # cVAR를 위한 알파

        S_T = self.S_t_input[-1,:,0] # T에서의 스팟

        # 스팟의 변화
        dS = self.S_t_input[1:,:,0] - self.S_t_input[0:-1,:,0]
        # dS = tf.reshape(dS,(time_steps,batch_size))

        # 2. RNN에서 사용할 S_t 준비 마지막 시간 단계 제거 (T에서 포트폴리오는 0임)
        S_t = tf.unstack(self.S_t_input[:-1,:,:], axis=0)

        ################ 빌드 ###############

        lstm = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(n)\
            for n in nodes])
        
        # 3. 상태는 0을 무시하고 마지막 실제 RNN 상태를 보유하는 텐서임.
        # 전략 텐서는 모든 셀의 출력을 보유함.
        self.strategy, state = tf.nn.static_rnn(lstm,S_t,initial_state=\
            lstm.zero_state(batch_size,tf.float32),dtype=tf.float32)
        self.strategy = tf.reshape(self.strategy, (time_steps-1, batch_size))

        # 4. 옵션 가격
        self.option = tf.maximum(S_T - self.K, 0)
        self.Hedging_PnL = -self.option + tf.reduce_sum(dS*self.strategy, \
            axis=0)
        
        # 5. 각 경로의 총 헤지 손익
        self.Hedging_PnL_Paths = -self.option + dS*self.strategy

        # 6. 주어진 신뢰수준 알파에 대한 CVaR 계산
        # 1-알파 최대 손실(상위 1-알파 음의 손익)을 취하고 평균을 계산함.
        
        CVaR, idx = tf.nn.top_k(-self.Hedging_PnL, tf.cast((1-self.alpha)*\
            batch_size, tf.int32))
        
        CVaR = tf.reduce_mean(CVaR)
        
        # 7. CVaR 최소화
        self.train = tf.train.AdamOptimizer().minimize(CVaR)
        self.saver = tf.train.Saver()
        self.modelname = name

    def _execute_graph_batchwise(self, paths, strikes, riskaversion, sess, \
        epochs=1, train_flag=False):
        # 1: 변수 초기화
        sample_size = paths.shape[1]
        batch_size= self.batch_size
        idx = np.arange(sample_size)
        start = dt.datetime.now()
        # 2: 모든 에폭에 걸쳐 반복
        for epoch in range(epochs):
            # 각 배치에 대한 헤징 손익 저장
            pnls = []
            strategies = []
            if train_flag:
                np.random.shuffle(idx)
            # 3: 관찰 반복
            for i in range(int(sample_size/batch_size)):
                indices = idx[i*batch_size : (i+1) * batch_size]
                batch = paths[:,indices,:]

                # 4: LSTM 훈련
                if train_flag: # 훈련 실행 및 손익과 전략 헤징
                    _, pnl, strategy = sess.run([self.train, self.Hedging_PnL,\
                        self.strategy], {self.S_t_input: batch,\
                        self.K : strikes[indices],\
                        self.alpha : riskaversion})
                        # 5: 평가 및 비훈련
                else:
                    pnl, strategy = sess.run([self.Hedging_PnL, self.strategy],\
                        {self.S_t_input: batch,\
                        self.K : strikes[indices], self.alpha : riskaversion})
                pnls.append(pnl)
                strategies.append(strategy)

            # 6 : 위험 회피 수준 알파를 고려한 옵션 가격 계산
            CVaR = np.mean(-np.sort(np.concatenate(pnls))\
                [:int((1-riskaversion)*sample_size)])
            # 7 : 훈련 단계에서 훈련 메트릭 반환
            if train_flag:
                if epoch % 10 == 0:
                    print('Time elapsed:', dt.datetime.now()-start)
                    print('Epoch',epoch,'CVaR',CVaR)
                    #Saving the model 모델 저장
                    self.saver.save(sess,"model.ckpt")
                
                    # 8: CVaR 및 기타 매개변수 변환
                    return CVaR, np.concatenate(pnls), np.concatenate(strategies,axis=1)

    # session input variable -> sess로 내가 임의 수정
    def training(self,paths,strikes,riskaversion,epochs,sess, init=True):
        if init:
            sess.run(tf.global_variables_initializer())
            # session
        self._execute_graph_batchwise(paths,strikes,riskaversion, sess,\
            epochs, train_flag=True)
    
    def predict(self,paths,strikes,riskaversion,session):
        return self._execute_graph_batchwise(paths,strikes,riskaversion,\
            session,1,train_flag=False)
    
    def restore(self,sess,checkpoint):
        self.saver.restore(sess,checkpoint)
