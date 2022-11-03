from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
#from cv2 import dft
import tensorflow as tf
import numpy as np

import base64
import matplotlib.pyplot as plt
from tensorflow.python.eager.monitoring import time
from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer

from utils import monte_carlo_paths

from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import py_environment, utils
from tf_agents.environments import tf_py_environment,tf_environment
from tf_agents.utils import common
from tf_agents.specs import array_spec
from tf_agents.trajectories import trajectory
from tf_agents.trajectories import time_step as ts
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer




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
    def __init__(self,df,balance):
        self.start_balance = balance #초기 잔고
        self.balance = balance # 100000
        self.df = df # ex : (31,1)
        self.amount = 0 

        # spaces
        # Actions : [buy,stay,sell]
        self._action_spec = array_spec.BoundedArraySpec(
          shape=(), dtype=np.int32, minimum=0, maximum=0, name='action')
        
        # Observations : [stock 주가 30일치]
        self._observation_spec = array_spec.BoundedArraySpec(shape=(5,), dtype=np.int32, minimum=0, name='observation')

        # episode
        #self._current_time_step = 0 # Timestamp 값임
        self.episode_step = 0
        self._episode_ended = False
        self.reward = 0
        self._total_cost = 0

        # state
        self._state = [self.reward,self._total_cost,self.balance,self.amount,self.episode_step]

    def action_spec(self): # 필수 return 타입 : types.NestedArraySpec:
        return self._action_spec 

    def observation_spec(self): # 필수 return 타입 : types.NestedArraySpec:
        return self._observation_spec

    def _reset(self): # 필수 return 타입 : ts.TimeStep
        self._episode_ended = False
        self.amount = 0
        self.episode_step = 0
        self.reward = 0
        self._total_cost = 0
        self.balance = self.start_balance
        self._state = [self.reward,self._total_cost,self.balance,self.amount,self.episode_step] # 변경 가능
        return ts.restart(np.array(self._state, dtype=np.int32)) #  초기화
    
    def getTotalValue(self):
        return float(self.balance + self.amount * self.df[self.episode_step])

    def _step(self, action): # 필수 return 타입 : ts.TimeStep
        """
            ### _state(observation)

            | Num | parameter              |
            |-----|------------------------| 
            | 0   | reward                 |
            | 1   | total_cost             |
            | 2   | balance                |
            | 3   | amount                 |
            | 4   | episode_step           |

        """
        self.ts_state()
        self.act(action) # action 진행 (action은 정수 0,1,2 중 하나)
        
        self._state = [self.reward,self._total_cost,self.balance,self.amount,self.episode_step]
        #print(self._state)

        self.isLoss()
        #self._state = [self.reward,self._total_cost] # 변경 가능
        if self.episode_step >= 30:# self._episode_ended or  or self._current_step == self._T: #  or self.get_loss() >= 10: #or self._current_step == self._T:
            self.reward = self.getTotalValue()
            print(f'step end : {self.episode_step}')
            #print(ts.termination(np.array(self._state, dtype=np.int32), self.reward))
            return ts.termination(np.array(self._state, dtype=np.int32), self.reward)
        elif self.isLoss() == True :
            self.reward = 0
            loss_rate = self._total_cost / self.start_balance
            print(f'loss out : {loss_rate} , episode : {self.episode_step}')
            return ts.termination(np.array(self._state, dtype=np.int32), self.reward)
        else:
            print(f'step continue : {self.episode_step}')
            self.reward = self.getTotalValue()
            self.episode_step +=1
            self._state = [self.reward,self._total_cost,self.balance,self.amount,self.episode_step]
            #print(self._state)

            # np.array([self._state]가 observation 값임)
            #return ts.transition(np.array([self._state], dtype=np.int32), self.reward,discount=1.0)
            #print(ts.transition(np.array(self._state, dtype=np.int32), self.reward,discount=1.0))
            return ts.transition(np.array(self._state, dtype=np.int32), self.reward,discount=1.0)


    # action 3개 백업(실제 실행되는 메소드 아님!!ㄴ)
    def _step_v2(self, action):
        cur_stock =  self.df[self.episode_step] # 현재 주가
        possible_buy_amount = int(self.balance/cur_stock) # 최대 구매 가능 수량
        possible_sell_amount = self.amount

        if action == 0: # 주식 사기
            random_percent = 0
            if possible_buy_amount != 0: random_percent = np.random.randint(0,possible_buy_amount) # 랜덤으로 (0~ 최대 구입가능 퍼센트에서 선택)
            self.amount += random_percent
            self.balance -= cur_stock * random_percent
            return

        elif action == 1: #주식 유지
            return

        elif action == 2: #주식 팔기
            if possible_sell_amount == 0:
                self.balance -= 1
            else:
                random_percent = np.random.randint(0,possible_sell_amount)
                self.amount -= random_percent
                self.balance += cur_stock * random_percent
            return
        else:
            raise  ValueError('`action` should be 0 or 1 or 2')

    # _state 를 파라미터에 적용 #
    def ts_state(self):
        self.reward = self._state[0]
        self._total_cost = self._state[1]
        self.balance = self._state[2]
        self.amount = self._state[3]
        self.episode_step = self._state[4]

    def isLoss(self):
        # 손실율 = (비용 - 현재주가 * (전량 매도)) / 현재 주가
        # if 손실율 > 0.1:
        #print(self._total_cost, self.start_balance)
        cur_stock =  self.df[self.episode_step]
        loss_rate = (self._total_cost - cur_stock * self.amount) / self.start_balance # 현재일 주가
        #print(f'loss rate : {loss_rate}, epdisode : {self.episode_step}')
        #print(f'{self._total_cost} | {self.start_balance}')
        if loss_rate > 0.1:
            return True

    def act(self,action):
        #print(f'step : {self.episode_step}')
        if self._episode_ended: # 다음 단계 준비
            return self.reset()

        #print(self.episode_step)

        cur_stock =  self.df[self.episode_step] # 현재 주가
        #self._state[2] => balance
        possible_buy_amount = int(self._state[2]/cur_stock) # 최대 구매 가능 수량
        possible_sell_amount = self._state[3]
        if action == 0:
            random_percent = np.random.randint(-possible_sell_amount,possible_buy_amount) # 랜덤으로 (0~ 최대 구입가능 퍼센트에서 선택)
            if random_percent >0 :
                self.amount += random_percent
                cost = cur_stock * random_percent
                self.balance -= cost
                self._total_cost += cost
                return
                #self.reward +=1
            elif random_percent < 0 :
            
                self.amount += random_percent
                
                cost = cur_stock * random_percent
                self.balance -= cost
                self._total_cost += cost
                return
        else:
            raise  ValueError('`action` should be 0 or 1 or 2')                


        '''
        if action == 0: # 주식 사기
            #print("buy")
            random_percent = np.random.randint(0,possible_buy_amount) # 랜덤으로 (0~ 최대 구입가능 퍼센트에서 선택)
            self.amount += random_percent
            cost = cur_stock * random_percent
            self.balance -= cost
            self._total_cost += cost
            self.reward +=1
            #print(f'buy item , cost : {cost} , total_ amount : {self.amount}, balance : {self.balance} , total_cost : {self._total_cost}')
            return
            # 현재 주가 * x = 비용
            # 총 비용 += 비용
            #pass
        elif action == 1: #주식 유지
            #print("stay")
            self.reward -= 1
            return
            #pass
        elif action == 2: #주식 팔기
            #print("sell")
            if possible_sell_amount == 0:
                random_percent = 0

            else:
                random_percent = np.random.randint(0,possible_sell_amount)
                self.amount -= random_percent
                
                profit = cur_stock * random_percent
                self.balance += profit
                self._total_cost -= profit
                #print(f'sell item , cost : {profit} , total_ amount : {self.amount}, balance : {self.balance} , total_cost : {self._total_cost}')
            self.reward -= 1 # 임시 추가
            # 현재 주가 * x = 비용
            # 총 비용 -= 비용
            # self._current[1] 주식이 1개 이상이면 팔기
            return
            #pass
        else:
            raise  ValueError('`action` should be 0 or 1 or 2')
        '''
        #self._episode_ended = True # 현 단계 종료
