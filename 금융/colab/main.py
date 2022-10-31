import datetime as dt
import numpy as np
import tensorflow as tf

class Agent(object):
    def __init__(self,STOCK_GRAPH,time_steps=30,strike=100,batch_size=10,features=1):
        self.batch_size = batch_size
        self.batch_GRAPH = STOCK_GRAPH # 입력받은 배치 그래프의 크기
        self.time_steps = time_steps
        self.STOCK_K = np.full((batch_size,),strike)
        self.STOCK_T = STOCK_GRAPH[-1,:,0] # (10000,)
        self.STOCK_dS = STOCK_GRAPH[1:,:,0] - STOCK_GRAPH[0:-1,:,0] # (30,10000)
        #self._option = tf.maximum(self.STOCK_T - self.STOCK_K,0)
        #self.delta = tf.reshape(self.delta,(time_steps-1,batch_size))
        #self._payoff = -self._option + tf.reduce_mean(self.STOCK_dS*self.delta,axis=0)
        #self._payoff_paths = -self._option + self.STOCK_dS*self.delta
        

    def _model(self,):
        alpha = 0.5
        input = tf.keras.layers.Input(shape=(30,1)) # (time_steps,feature)
        
        output,state_h,state_c = tf.keras.layers.LSTM(64,return_state=True)(input)
        output = tf.keras.layers.Dense(32,activation='tanh')(output)
        output = tf.keras.layers.Dense(1)(output)
        
        strategy = tf.reshape(output,(self.time_steps-1,self.batch_size))
        strategy = tf.cast(strategy,tf.float64)
        print(strategy.shape)
        option = tf.maximum(self.STOCK_T - self.STOCK_K,0)
        print(option.shape)
        Hedging_PnL = - option + tf.reduce_sum(self.STOCK_dS * strategy, axis=0)
        print(Hedging_PnL.shape)
        Hedging_PnL_paths = - option + self.STOCK_dS * strategy #(10000,) + (30,10000)
        print(Hedging_PnL_paths.shape)

        CVaR, idx = tf.math.top_k(-Hedging_PnL,k=tf.cast((1-alpha)* self.batch_size, tf.int32)) # must set to int32 !!
        CVaR = tf.reduce_mean(-Hedging_PnL)
        model = tf.keras.Model(inputs=input,outputs=[Hedging_PnL,CVaR])
        #model.summary()
        return model

    def _train(self,paths,strikes,riskaversion,epochs,train_flag=False):
        sample_size = paths.shape[1]
        batch_size = self.batch_size
        idx = np.arange(sample_size)

        for epoch in range(epochs):
            pnls = []
            strategies = []
            if train_flag:
                np.random.shuffle(idx)
            for i in range(int(sample_size/batch_size)):
                indices = idx[i*batch_size:(i+1)*batch_size]
                batch = paths[:,indices,:]
                if train_flag:
                    model = self._model()
                    model.compile(loss='mse',optimizer='adam')
                    #_, pnl, strategy = model.fit() -> 여기서 훈련 결과 값 받아서

                pnls.append(pnl) # payoff 값에 추가
                strategies.append(strategy) # delta 값들에 추가
        
        CVaR = np.mean(-np.sort(np.concatenate(pnls))[:int((1-riskaversion)*sample_size)]) # CVaR 결과값

        return CVaR, np.concatenate(pnls), np.concatenate(strategies,axis=1)


# STOCK_GRAPH,time_steps=30,strike=100,batch_size=10,features=1
Agent(S_t_input, strikes[indices], riskaversion)                  