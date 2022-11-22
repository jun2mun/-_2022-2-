import numpy as np
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


    def __init__(self, df, balance=100000):
        super(TradeEnv, self).__init__()
        # Actions : [buy,stay,sell]
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        # Observations : [stock 주가 30일치]
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        self.df = df  # 주가 데이터
        self.start_balance = balance
        self.balance = balance
        self.amount = 0
        self.current_step = 0
        self.reward = 0

    def reset(self, balance=100000):
        self.balance = balance
        self.amount = 0
        self.current_step = 0
        self.reward = 0
        return self._get_obs()

    def _get_obs(self):
        return np.array([self.df[self.current_step], self.balance, self.amount, self.reward], dtype=np.float32)

    def action_method(self, action):

        cur_stock = self.df[self.current_step]  # 현재 주가
        possible_buy_amount = int(self.balance / cur_stock)  # 최대 구매 가능 수량
        possible_sell_amount = self.amount  # 최대 판매 가능

        if action < 0: #주식 판매
            sell_amount = possible_sell_amount * action
            self.balance += sell_amount * cur_stock
            self.amount -= sell_amount * cur_stock
        elif action >= 0: #주식 구매
            buy_amount = possible_buy_amount * action
            self.balance -= buy_amount * cur_stock
            self.amount += buy_amount * cur_stock

    def _get(self):
        return self.current_step

    def step(self, action):
        done = False
        self.action_method(action)

        if self.isLoss():
            done = True
        else:
            self.reward += 1

        self.current_step += 1
        if self.current_step >= 30:
            self.current_step =0
            print("================================================self.curre")
            done = True

        return self._get_obs(), self.reward, done, False

    def render(self):
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')

    def getTotalValue(self):
        return self.amount * self.df[self.current_step] + self.balance

    def isLoss(self):
        # 손실율 = (비용 - 현재주가 * (전량 매도)) / 현재 주가
        # if 손실율 > 0.1:
        loss_rate = (self.getTotalValue() - self.start_balance[self.current_step]) / self.start_balance[self.current_step]  # 현재일 주가
        if loss_rate <= -0.02:
            return True
        else: return False
