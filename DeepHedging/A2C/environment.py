import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import sys

DATA_PATH = './Input/Data/Stocks'

class TradeEnv():
    def reset(self):
        self.data = self.gen_universe()
        self.pos = 0
        self.game_length = self.data.shape[0]
        self.returns = []

        # return first state
        return self.data[0, :-1, :]

    def step(self, allocation):
        ret = np.sum(allocation * self.data[self.pos, -1, :])
        self.returns.append(ret)
        mean = 0
        std = 1
        if len(self.returns) >= 20:
            mean = np.mean(self.returns[-20:])
            std = np.std(self.returns[-20:]) + 0.0001
        sharpe = mean / std

        if (self.pos + 1) >= self.game_length:
            return None, sharpe, True, {}
        else:
            self.pos += 1
            return self.data[self.pos, :-1, :], sharpe, False, {}

    def gen_universe(self):
        stocks = os.listdir(DATA_PATH)
        stocks = np.random.permutation(stocks)
        frames = []
        idx = 0
        while len(frames) < 100:
            try:
                stock = stocks[idx]
                frame = pd.read_csv(os.path.join(DATA_PATH, stock), index_col='Date')
                frame = frame.loc['2005-01-01':].Close
                frames.append(frame)
            except:  # catch *all* exceptions
                e = sys.exc_info()[0]
            idx += 1

        df = pd.concat(frames, axis=1, ignore_index=False)
        df = df.pct_change()
        df = df.fillna(0)
        batch = df.values
        episodes = []
        for i in range(batch.shape[0] - 101):
            eps = batch[i:i + 101]
            episodes.append(eps)
        data = np.stack(episodes)
        assert len(data.shape) == 3
        assert data.shape[-1] == 100
        return data