import numpy as np
from gym import Env
from gym import spaces


class TradeEnv(Env):

    start_balance = 1000000

    def __init__(self, df):
        self.id = 'default'
        self.balance = TradeEnv.start_balance  # 100000
        self.df = df  # ex : (31,1)

        low = np.array([0, 0], dtype=np.int32)
        high = np.array([np.iinfo(np.int32).max,np.iinfo(np.int32).max], dtype=np.int32)
        # self.action_space = spaces.Box(low=0 , high= 2 , shape=(1,), dtype=np.int32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        # Observations
        self.observation_space = spaces.Box(
            low=low, high=high, dtype=np.int32)

    def reset(self):

        self.balance = TradeEnv.start_balance  # 100000
        self.amount = 0
        self.current_step = 0 # day
        self.reward = 0

        #return np.array([self.df[self.current_step], self.balance, self.amount])
        return np.array([self.df[self.current_step],self.balance, self.amount])

    def step(self, action):
        self.action_method(action)
        
        self.reward = self.reward_func()
        terminated = False

        if self.current_step == len(self.df) -2:
            terminated = True
            print(f'reward is {self.reward}')
        
        self.current_step +=1
       
        #return np.array([self.df[self.current_step], self.balance, self.amount]), self.reward, terminated, False
        return np.array([self.df[self.current_step],self.balance, self.amount]), self.reward, terminated, False


    def action_method(self, action):

        if action < 0: #주식 판매
            self.sell(action)

        elif action > 0: #주식 구매
            self.buy(action)

        elif action == 0:
            print('hold')
            pass

    def sell(self,amount):
        cur_stock = self.df[self.current_step]  # 현재 주가
        possible_sell_amount = int(self.balance / cur_stock)  # 최대 구매 가능 수량

        if possible_sell_amount < amount:
            return False
        else:
            sell_amount = amount #random_percent
            self.balance += sell_amount * cur_stock
            self.amount -= sell_amount
            return True

    def buy(self,amount):
        cur_stock = self.df[self.current_step]  # 현재 주가
        possible_buy_amount = int(self.balance / cur_stock)  # 최대 판매 가능

        if possible_buy_amount < amount:
            return False
        else:
            buy_amount = amount #random_percent
            self.balance -= buy_amount * cur_stock
            self.amount += buy_amount
            return True

    def reward_func(self): # 수익율
        self.balance += self.amount * self.df[self.current_step]
        reward = ((self.balance - TradeEnv.start_balance) / TradeEnv.start_balance) * 100
        return reward


    ####################################################################
    def setid(self,id):
        self.id = id

    def render(self):
        return self.current_step