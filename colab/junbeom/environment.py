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
    | 0   | 주식 사기(-x% ~ +x%)    |

    ### Observation Space

    | Num | parameter              |
    |-----|------------------------| 
    | 0   | reward                 |
    | 1   | total_cost             |
    | 2   | balance                |
    | 3   | amount                 |
    | 4   | episode_step           |
    
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
        # Actions : [buy x %]
        self._action_spec = array_spec.BoundedArraySpec(
          shape=(), dtype=np.int32, minimum=0, maximum=2, name='action')
        
        # Observations 
        self._observation_spec = array_spec.BoundedArraySpec(shape=(4,), dtype=np.float64, minimum=0, name='observation')

        # episode
        #self._current_time_step = 0 # Timestamp 값임
        self.episode_step = 0
        self.reward = 0
        self._total_cost = 0

        
    def action_spec(self): # 필수 return 타입 : types.NestedArraySpec:
        return self._action_spec 

    def observation_spec(self): # 필수 return 타입 : types.NestedArraySpec:
        return self._observation_spec

    # 초기화 함수
    def _reset(self): # 필수 return 타입 : ts.TimeStep
        self.amount = 0
        self.episode_step = 0
        self.reward = 0
        self._total_cost = 0
        self.balance = self.start_balance
        return ts.restart(np.array([self._total_cost, self.balance, self.amount, self.episode_step], dtype=np.float64)) #  초기화
    
    # 타입스텝 진행
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
        
        # action 실행
        self.act(action) # action 진행 (action은 정수 0,1,2 중 하나)
        
        # # Loss 확인
        # self.isLoss()
        
        # 에피소드 step 증가
        self.episode_step +=1

        # 보상 함수 실행
        self.calc_reward()

        # when episode end
        if self.episode_step == 30:
            #print(f'step end : {self.episode_step}')
            return ts.termination(np.array([self._total_cost,self.balance,self.amount,self.episode_step], dtype=np.float64), self.reward)
        
        # when episode not end
        else:
            self.reward = 1
            return ts.transition(np.array([self._total_cost,self.balance,self.amount,self.episode_step], dtype=np.float64), self.reward,discount=1.0)


    # 손실율 초과 시 terminate 시키는 메소드
    def calc_reward(self):
        # 손실율 = (비용 - 현재주가 * (전량 매도)) / 현재 주가
        cur_stock =  self.df[self.episode_step]
        loss_rate = (self._total_cost - cur_stock * self.amount) / self.start_balance # 현재일 주가
        if loss_rate > 0.02:
            self.reward = -(self._total_cost - cur_stock * self.amount)
        else:
            self.reward = -(self._total_cost - cur_stock * self.amount)

    def act(self,action):
        self.reward = 0
        cur_stock =  self.df[self.episode_step] # 현재 주가
        possible_buy_amount = int(self.balance/cur_stock) # 최대 구매 가능 수량
        possible_sell_amount = self.amount # 최대 판매 가능
        
        if action == 0: # 주식 사기
            random_percent = np.random.randint(0,possible_buy_amount+1) # 랜덤으로 (0~ 최대 구입가능 퍼센트에서 선택)
            random_percent = 1
            self.amount += random_percent
            cost = cur_stock * random_percent
            self.balance -= cost
            self._total_cost += cost
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
                self._total_cost -= profit
            return
        else:
            raise  ValueError('`action` should be 0 or 1 or 2')
        