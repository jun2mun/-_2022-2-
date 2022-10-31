from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
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

class HedgeENV(py_environment.PyEnvironment):
    
    """
    ### Action Space

    | Num | Action                 |
    |-----|------------------------| 
    | 0   | 주식 팔기               |
    | 1   | 가만히 있기             |
    | 2   | 주식 사기               |

    ### Observation Space

    | Num | Observation           | Min                 | Max               |
    |-----|-----------------------|---------------------|-------------------|
    | 0   | Loss                  | -0.10               | 0.10              |
    
    """

    init_current = [1000, 0]
    unit = 1

    def __init__(self,S,T):
        
        # Actions : [buy,stay,sell]
        self._action_spec = array_spec.BoundedArraySpec(
          shape=(), dtype=np.int32, minimum=0, maximum=2, name='action')
        
        # Observations : ?
        self._observation_spec = array_spec.BoundedArraySpec(shape=(1,), dtype=np.int32, minimum=0, name='observation')
        
        # 초기화
        self._S = S  # 주가데이터
        self._T = T # 만기 날짜
        
        self._state = 0 # 추후에 생각해보기
        self._episode_ended = False

        self._current_step = 0 # 현재 단계
        '''
        self._current[0] =
        self._current[1] = 
        '''
        self._current = HedgeENV.init_current # 초기 상태
        
        self.reward = 0

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._state = 0  # 추후에 생각해보기
        self._episode_ended = False
        return ts.restart(np.array([self._state], dtype=np.int32))
        

    def isLoss(self):
        # 손실율 = (비용 - 현재주가 * (전량 매도)) / 현재 주가
        # if 손실율 > 0.1:
        # terminate
        pass

    def get_loss(self):
        # total_cost = '기초자산' + '주식개수' * '주식 현재가'
        total_cost = self._current[0] + self._current[1] * self._S[self._current_step]
        f_cost = HedgeENV.init_current[0] # 초기기초자산
        return abs(f_cost - total_cost) / f_cost * 100 # 변화율 (x)%

    def act(self,action):
        if self._episode_ended: # 다음 단계 준비
            return self.reset()

        if action == 0: # 주식 팔기
            # 현재 주가 * x = 비용
            # 총 비용 -= 비용
            # self._current[1] 주식이 1개 이상이면 팔기
            
            if self._current[1] > 0 :
                self._current[0] += self._S[self._current_step] * HedgeENV.unit # 주식 보유하고 있으면 판매
            #print("sell")
        elif action == 1: #주식 유지
            #print("stay")
            pass
        elif action == 2: #주식 사기
            #print("buy")
            # 현재 주가 * x = 비용
            # 총 비용 += 비용
            if self._current[0] >= self._S[self._current_step] * HedgeENV.unit :
                self._current[0] -= self._S[self._current_step] * HedgeENV.unit
                self._current[1] += HedgeENV.unit
        else:
            raise  ValueError('`action` should be 0 or 1 or 2')

        self._episode_ended = True # 현 단계 종료


    def _step(self, action):
        #print("1",self._episode_ended)

        self.act(action)
        

        #print("3",self._episode_ended,self.get_loss(),self._current_step,self._T)
        if self._episode_ended: #  or self.get_loss() >= 10: #or self._current_step == self._T:
            #print("1")
            return ts.termination(np.array([self._state], dtype=np.int32), self.reward)
            
        else:
            #print("2")
            self.reward += 1
            self._current_step += 1
            return ts.transition(np.array([self._state], dtype=np.int32), self.reward, discount=1.0)

"""
ts.transition
TimeStep(
{'discount': array(1., dtype=float32),
 'observation': array([0]),
 'reward': array(0., dtype=float32),
 'step_type': array(1)})
"""