from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tf_agents.environments import py_environment, utils
from tf_agents.trajectories import time_step as ts
from tf_agents.specs import array_spec
import gym
from gym import Env
from gym import spaces

class TradeEnv(Env):
    
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
    metadata = {'render.modes': ['human']}
    
    def __init__(self,df,balance=100000):
        super(TradeEnv,self).__init__()        
        # Actions : [buy,stay,sell]
        self.action_space = spaces.Box(
            low=0 , high= 2 , shape=(1,), dtype=np.int32
        )
        # Observations : [stock 주가 30일치]
        self.observation_space = spaces.Box(
            low=-1.0 , high= 1.0 , shape=(4,), dtype=np.float32
        )
        #################### 테스트 변수 #####################
        self.df = df # 주가 데이터
        
    def reset(self,balance=100000):
        self.balance = balance
        self.amount = 0
        self.total_cost = 0
        self.current_step = 0
        return self._get_obs()

    def _get_obs(self):
        return np.array([self.df[self.current_step],self.total_cost,self.balance,self.amount],dtype=np.float32)

    def action_method(self,action):

        self.reward = 0
        cur_stock =  self.df[self.current_step] # 현재 주가
        possible_buy_amount = int(self.balance/cur_stock) # 최대 구매 가능 수량
        possible_sell_amount = self.amount # 최대 판매 가능
        
        if action == 0: # 주식 사기
            if possible_buy_amount == -1:
                return
            random_percent = np.random.randint(0,possible_buy_amount+1) # 랜덤으로 (0~ 최대 구입가능 퍼센트에서 선택)
            random_percent = 1
            self.amount += random_percent
            cost = cur_stock * random_percent
            self.balance -= cost
            self.total_cost += cost
            return

        elif action == 1: #주식 유지
            return

        elif action == 2: #주식 팔기
            if possible_sell_amount == 0:
                random_percent = 0

            else:
                random_percent = np.random.randint(0,possible_sell_amount)
                self.amount -= random_percent
                
                profit = cur_stock * random_percent
                self.balance += profit
                self.total_cost -= profit
            return
        else:
            raise  ValueError('`action` should be 0 or 1 or 2')

    def _get(self):
        return self.current_step

    def step(self,action):
        done = False
        self.action_method(action)

        self.current_step +=1

        
        if self.isLoss():
            self.reward -= 0.6
        else:
            self.reward += 1

        if self.current_step >= 30:
            self.current_step = 0
            done = True


        return self._get_obs(), self.reward, done, False

    def render(self):
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        
    def isLoss(self):
        # 손실율 = (비용 - 현재주가 * (전량 매도)) / 현재 주가
        # if 손실율 > 0.1:
        self.loss_Rate = self.total_cost / self.balance # 현재일 주가
        if self.loss_Rate > 0.02:
            return True
        # terminate
        pass
