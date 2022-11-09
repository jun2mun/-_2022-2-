from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tf_agents.environments import py_environment
from tf_agents.trajectories import time_step as ts
from tf_agents.specs import array_spec

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

 ### _state(observation)

 | Num | parameter              |
 |-----|------------------------|
 | 0   | reward                 |
 | 1   | total_cost             |
 | 2   | balance                |
 | 3   | amount                 |
 | 4   | episode_step           |
 """

class TradeEnv(py_environment.PyEnvironment):

    def __init__(self,df, balance):
        self.start_balance = balance #시작 자본금
        self.current_balance = balance #현 자본금
        self.amount = 0  # 주식 보유량
        self.df = df #주가 데이터
        self.reward = 0 #액션에 따른 보상
        self.episode_step = 0 #현 episode 진행 단계
        self.count = 0 #loss count

        # spaces
        # Actions : [buy,stay,sell]
        self._action_spec = array_spec.BoundedArraySpec(
          shape=(), dtype=np.float64, minimum=0.0, maximum=2.0, name='action')
        
        # Observations : [stock 주가 30일치]
        self._observation_spec = array_spec.BoundedArraySpec(shape=(3,), dtype=np.float64, minimum=0, name='observation')

    def action_spec(self): # 필수 return 타입 : types.NestedArraySpec:
        return self._action_spec 

    def observation_spec(self): # 필수 return 타입 : types.NestedArraySpec:
        return self._observation_spec

    def _reset(self): # 필수 return 타입 : ts.TimeStep
        self.balance = self.start_balance
        self.amount = 0
        self.reward = 0
        self.episode_step = 0

        return ts.restart(np.array([self.balance, self.amount, self.episode_step], dtype=np.float64)) #  초기화

    def _step(self, action): #필수 return 타입 : ts.TimeStep
        print(f'action : {action}')
        self.action(action) #action 진행 (action type : float[0~2])

        if self.isLoss() : #penatly 부과 조건 (기준 손실율 초과)
            self.putPenalty()
        else:
            self.count = 0  # self.count - 0으로 초기화
            self.reward += self.getTotalValue() * 0.01

        if self.episode_step >= 30: #episode 종료 조건 (만기일 도달)
            print(f'step end : {self.episode_step}')
            return ts.termination(np.array([self.balance, self.amount, self.episode_step], dtype=np.float64), self.reward + self.start_balance * 0.2)
        else:
            self.episode_step +=1
            return ts.transition(np.array([self.balance, self.amount, self.episode_step], dtype=np.float64), self.reward ,discount=1.0)

    def getTotalValue(self):
        return float(self.balance + self.amount * self.df[self.episode_step])

    def isLoss(self):
        # 손실율 = (비용 - 현재주가 * (전량 매도)) / 현재 주가
        # if 손실율 > 0.1:
        cur_stock =  self.df[self.episode_step]
        loss_rate = (self.getTotalValue() - self.start_balance) / self.start_balance
        if loss_rate <= -0.02:
            return True

    def getChangeRate(self):
        return self.df[self.episode_step] - self.df[self.episode_step -1]
    def putPenalty(self):
        self.count += 1
        self.reward -= self.getChangeRate()
        if self.count >= len(self.df) * 0.2: # episode 종료 조건 2 (기준 손해 초과)
            self.reward += self.getTotalValue()
            return ts.termination(np.array([self.balance, self.amount, self.episode_step], dtype=np.float64),
                                  self.reward / 2)


    def action(self,action):
        cur_stock =  self.df[self.episode_step] # 현재 주가
        criteria1 = 0.7
        criteria2 = 1.3

        if action < criteria1 : #주식 판매
            sell_amount = self.amount * action
            self.amount -= sell_amount
            self.balance += cur_stock * sell_amount
        elif action > criteria2 : #주식 구매
            possible_buy_amount = self.balance / cur_stock
            buy_amount = possible_buy_amount * (action - 1)
            self.amount += buy_amount
            self.balance -= buy_amount * cur_stock
        else : #그대로 유지
            pass


