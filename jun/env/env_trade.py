import numpy as np
from gym import Env
from gym import spaces


class TradeEnv(Env):

    start_balance = 1000000

    def __init__(self, df):
        self.id = 'default'
        self.balance = TradeEnv.start_balance  # 100000
        self.df = df  # ex : (31,1)
        self.amount = 0
        self.current_step = 0
        self.reward = 0


        # self.action_space = spaces.Box(low=0 , high= 2 , shape=(1,), dtype=np.int32)
        self.action_space = spaces.Box(low=-1.00, high=1.00, shape=(1,), dtype=np.float32)

        # Observations
        self.observation_space = spaces.Box(
            low=-TradeEnv.start_balance, high=TradeEnv.start_balance, shape=(3,), dtype=np.float32)

    def reset(self):
        self.balance = TradeEnv.start_balance  # 100000
        self.amount = 0
        self.current_step = 0
        self.reward = 0
        self.action_method(0)

        return np.array([self.df[self.current_step], self.balance, self.amount])

    def step(self, action):
        self.action_method(action)
        #print(f'current id =  {self.id} , current_step = {self.current_step}')
        
        self.reward += self.getTotalValue()
        
        terminated = self.isLoss()

        if self.current_step == len(self.df) -1:
            terminated = True
        
        self.current_step +=1
       
        return np.array([self.df[self.current_step], self.balance, self.amount]), self.reward, terminated, False

    def action_method(self, action):
        #random_percent = float(np.random.rand(1))
        cur_stock = self.df[self.current_step]  # 현재 주가
        possible_buy_amount = int((self.balance / cur_stock) / 5)  # 최대 구매 가능 수량
        possible_sell_amount = int(self.amount / 5)  # 최대 판매 가능

        if action < 0: #주식 판매
            print('sell')
            sell_amount = possible_sell_amount * action #random_percent
            self.balance += sell_amount * cur_stock
            self.amount -= sell_amount

        elif action > 0: #주식 구매
            print('buy')
            buy_amount = possible_buy_amount * action #random_percent
            self.balance -= buy_amount * cur_stock
            self.amount += buy_amount
            print(buy_amount,self.balance,self.amount,cur_stock,self.current_step)
        
        elif action == 0:
            print('hold')
            pass

    def setid(self,id):
        self.id = id

    def render(self):
        return self.current_step

    def getTotalValue(self):
        return (self.amount * self.df[self.current_step] + self.balance) - TradeEnv.start_balance

    def isLoss(self):
        # 손실율 = (비용 - 현재주가 * (전량 매도)) / 현재 주가
        # if 손실율 > 0.1:
        loss_rate = (self.getTotalValue() - self.start_balance) / self.start_balance  # 현재일 주가
        if loss_rate <= -0.02:
            print('loss')
            return True
        else:
            return False


