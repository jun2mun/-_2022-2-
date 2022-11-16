import numpy as np

class RandomTrader():
    def get_action(self):
        action = np.random.rand(100) * 2 - 1
        action = action * (np.abs(action) / np.sum(np.abs(action)))
        return action