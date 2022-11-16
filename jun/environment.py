from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tf_agents.environments import py_environment, utils
from tf_agents.trajectories import time_step as ts
from tf_agents.specs import array_spec


class TradeEnv(py_environment.PyEnvironment):
    
    """
    ### Action Space

    | Num | Action                 |
    |-----|------------------------| 
    | 0   | 주식 사기               |
    | 1   | 가만히 있기             |
    | 2   | 주식 팔기               |

    ### Observation Space

    | Num | Observation           | Min                 | Max               |
    |-----|-----------------------|---------------------|-------------------|
    | 0   | Loss                  | -0.10               | 0.10              |
    
    ### portfolio 데이터

    | Num | parameter              |
    |-----|------------------------| 
    | 0   | stock dataframe(30)    |
    | 1   | epsilon                |
    | 2   |            |
    
    
    """

    def __init__(self,df,balance=100000):
        
        # Actions : [buy,stay,sell]
        self._action_spec = array_spec.BoundedArraySpec(
          shape=(), dtype=np.int32, minimum=0, maximum=2, name='action')
        
        # Observations : [stock 주가 30일치]
        self._observation_spec = array_spec.BoundedArraySpec(shape=(1,), dtype=np.int32, minimum=0, name='observation')
        
        # 초기화
        self._state = 0 # 추후에 생각해보기 -> 
        self._episode_ended = False
        
        self.reward = 0

        #################### 테스트 변수 #####################
        self.df = df # 주가 데이터
        self.balance = balance # 현재 잔고
        self.cost = 0 # 사용한 비용
        self.loss_Rate = 0 # 손실율(최악인 경우)

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._state = 0  # 추후에 생각해보기
        self._episode_ended = False
        self.reward = 0
        #self.balance
        return ts.restart(np.array([self._state], dtype=np.int32))
        

    def isLoss(self):
        # 손실율 = (비용 - 현재주가 * (전량 매도)) / 현재 주가
        # if 손실율 > 0.1:
        self.loss_Rate = self.cost / self.balance # 현재일 주가
        if self.loss_Rate > 0.1:
            return True
        # terminate
        pass

    def act(self,action):
        if self._episode_ended: # 다음 단계 준비
            return self.reset()

        if action == 0: # 주식 사기
            # 현재 주가 * x = 비용
            # 총 비용 -= 비용
            # self._current[1] 주식이 1개 이상이면 팔기
            pass
        elif action == 1: #주식 유지
            #print("stay")
            pass
        elif action == 2: #주식 팔기
            #print("buy")
            # 현재 주가 * x = 비용
            # 총 비용 += 비용
            pass
        else:
            raise  ValueError('`action` should be 0 or 1 or 2')

        self._episode_ended = True # 현 단계 종료


    def _step(self, action):

        self.act(action) # action 진행
        


        if self._episode_ended or self.isLoss() == True :# or self._current_step == self._T: #  or self.get_loss() >= 10: #or self._current_step == self._T:

            return ts.termination(np.array([self._state], dtype=np.int32), self.reward)
            
            
        else:
            self.reward += 1
            #self._current_step += 1
            #print(ts.transition(np.array([self._state], dtype=np.int32), self.reward, discount=1.0))
            # np.array([self._state]가 observation 값임)
            #return ts.transition(self.reward, reward=2,discount=1.0)
            return ts.transition(np.array([self._state], dtype=np.int32), self.reward,discount=1.0)