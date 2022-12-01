from dataclasses import dataclass

import numpy as np
import gym
from gym import spaces
from typing import Union

highSpace = np.array(
    [
        10.0,  # OPEN
        10.0,  # HIGH
        10.0,  # LOW
        10.0,  # CLOSE
        3.0,  # VOLUME
        3.0,  # KOSPI
        3.0,  # SP500
        3.0,  # MACD
        3.0,  # RSi
        3.0,  # BOLLANGER
        3.0,  # ATR

    ],
    dtype=np.float32,
)

lowSpace = np.array(
    [
        0.0,  # OPEN
        0.0,  # HIGH
        0.0,  # LOW
        0.0,  # CLOSE
        0.0,  # VOLUME
        0.0,  # KOSPI
        0.0,  # SP500
        0.0,  # MACD
        0.0,  # RSi
        0.0,  # BOLLANGER
        0.0,  # ATR
    ],
    dtype=np.float32,
)

@dataclass
class State:
    open : float = None
    high : float = None
    low: float = None
    close: float = None
    volume: float = None
    kospi: float = None
    sp500: float = None
    macd: float = None
    rsi : float = None
    bollinger: float = None
    atr: float = None

class TradeEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):

    start_balance = 10000000

    #df 형식 : ['Open', 'High', 'Low', 'Close', 'Volume', 'Kospi', 'Sp500', 'Macd','Rsi', 'Bollinger', 'Atr', 'Price']
    def __init__(self, df):
        self.balance = TradeEnv.start_balance  # 100000
        self.prices = df[:, -1]
        self.df = np.delete(df, -1, axis=1)  # ex : (T,11)
        self.amount = 0
        self.current_step = 0
        self.reward = 0

        # self.action_space = spaces.Box(low=0 , high= 2 , shape=(1,), dtype=np.int32)
        self.action_space = spaces.Discrete(3)

        # Observations (11,1)
        self.observation_space = spaces.Box(
            low=lowSpace, high=highSpace, dtype=np.float32)

    def reset(self):
        self.balance = TradeEnv.start_balance  # 100000
        self.amount = 0
        self.current_step = 0
        self.reward = 0
        self.action_method(1)

        return self.df[0]

    def step(self, action):
        self.action_method(action)
        self.reward = self.getEarningRate()
        self.current_step += 1

        terminated = self.isLoss()
        if self.current_step == len(self.df) -1:
            terminated = True

        return self.df[self.current_step], self.reward, terminated, False

    def action_method(self, action):
        cur_stock = self.prices[self.current_step]  # 현재 주가
        # random_percent = float(np.random.rand(1))
        # possible_buy_amount = int((self.balance / cur_stock) / 5)  # 최대 구매 가능 수량
        # possible_sell_amount = int(self.amount / 5)  # 최대 판매 가능

        if action == 0: #주식 판매
            # sell_amount = possible_sell_amount * random_percent
            sell_amount = 10
            if(self.amount >= sell_amount):
                self.balance += sell_amount * cur_stock
                self.amount -= sell_amount
        elif action == 1: #주식 구매
            # buy_amount = possible_buy_amount * random_percent
            buy_amount = 10
            if(self.balance >= cur_stock * buy_amount):
                self.balance -= buy_amount * cur_stock
                self.amount += buy_amount
        elif action == 2: #그대로 유지
            pass


    def getTotalValue(self):
        return self.amount * self.prices[self.current_step] + self.balance


    def getEarningRate(self):
        return ((self.getTotalValue() - self.start_balance) / self.start_balance) * 100

    def isLoss(self):
        # 손실율 = (비용 - 현재주가 * (전량 매도)) / 현재 주가
        # if 손실율 > 0.1:
        earn_rate = self.getEarningRate()
        if earn_rate <= -15:
            return True
        else:
            return False


