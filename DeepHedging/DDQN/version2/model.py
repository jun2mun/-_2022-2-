from DDQN.version2.env_trade import TradeEnv
from DDQN.version2.ddqn import DQNAgent
from DDQN.utils.StockUtils import getStockDataToTrain
import numpy as np
import matplotlib.pyplot as plt


save_location = "./save/"

class DDqnModel:

    def __init__(self):
        self.T = None #만기일
        self.S = None #주가정보
        self.update_interval = 10

    def setTrainSet(self, STOCK_CODE, start_date, end_date):
        self.S = getStockDataToTrain(STOCK_CODE, start_date,end_date)
        self.T = len(self.S)

    def train(self, num_episode, batch_size, update_interval, save_interval, plt_interval,
              is_load=False, load_file_name=None, save_file_name="hedge"):
        # Train environment 구성
        env = TradeEnv(self.S)
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        agent = DQNAgent(state_size, action_size)
        history = []
        earn_history = []


        # agent.load("./save/000720.KS_1000-ddqn.h5")

        if (is_load):  #모델 로드 여부
            agent.load(load_file_name)

        for e in range(num_episode):
            state = env.reset()
            state = np.reshape(state, [1, state_size])
            done = False

            for time in range(self.T):
                action = agent.act(state)
                print("action : {}".format(action))

                next_state, reward, done, _ = env.step(action)

                next_state = np.reshape(next_state, [1, state_size])
                agent.memorize(state, action, reward, next_state, done)
                state = next_state

                if done:
                    agent.update_target_model()
                    print("episode: {}/{}, score: {}, e: {:.2}"
                          .format(e, num_episode, time, agent.epsilon))
                    history.append(time)
                    earn_history.append(reward)
                    break

                if len(agent.memory) > batch_size:
                    agent.replay(batch_size)

            if (e + 1) % plt_interval == 0:
                interval = 30
                plt_hist = np.array(history).reshape(-1,interval)
                plt.plot(np.arange(0, len(history), interval), plt_hist.mean(axis=1), 'b')
                plt.xlabel('Episode')
                plt.ylabel('time')
                plt.show()

                plt_hist = np.array(earn_history).reshape(-1, interval)
                plt.plot(np.arange(0, len(earn_history), interval), plt_hist.mean(axis=1), 'r')

                plt.xlabel('Episode')
                plt.ylabel('earn rate')
                plt.show()

            if (e + 1) % save_interval == 0:
                agent.save(save_location + save_file_name + "_{}-ddqn.h5".format(e + 1))


        self.predictWithPrev(agent=agent)


    def predictWithPrev(self,agent):
        # Train environment 구성
        env = TradeEnv(self.S)
        state_size = env.observation_space.shape[0]
        buy_history = []
        sell_history = []

        state = env.reset()
        state = np.reshape(state, [1, state_size])

        for time in range(self.T):
            action = agent.predict(state)
            if (action == 0):  # 주식 판매
                sell_history.append(time)
            elif (action == 1):  # 주식 구매
                buy_history.append(time)

            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            state = next_state

            print("Time: {}/{}, reward: {}, action:{}"
                  .format(time, self.T, reward, action))

            if done: break;
        def setStockPrice(history):
            stockY = []
            for day in history:
                stockY.append(self.S[day][-1])
            return stockY

        plt.xlabel('time')
        plt.ylabel('price')
        plt.plot(np.arange(0, len(self.S)), self.S[:,-1])
        plt.plot(np.arange(0, len(buy_history)), setStockPrice(buy_history), 'bo')
        plt.plot(np.arange(0, len(sell_history)), setStockPrice(sell_history), 'ro')
        plt.show()


    def predict(self, load_file_name):
        # Train environment 구성
        env = TradeEnv(self.S)
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        buy_history = []
        sell_history = []

        agent = DQNAgent(state_size, action_size)
        agent.load(load_file_name)
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        for time in range(self.T):
            action = agent.predict(state)
            if (action == 0):  # 주식 판매
                sell_history.append(time)
            elif (action == 1):  # 주식 구매
                buy_history.append(time)

            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            state = next_state

            print("Time: {}/{}, reward: {}, action:{}"
                  .format(time, self.T, reward, action))

            if done: break;

        def setStockPrice(history):
            stockY = []
            for day in history:
                stockY.append(self.S[day][-1])
            return stockY

        plt.xlabel('time')
        plt.ylabel('price')
        plt.plot(np.arange(0, len(self.S)), self.S[:, -1])
        plt.plot(np.arange(0, len(buy_history)), setStockPrice(buy_history), 'bo')
        plt.plot(np.arange(0, len(sell_history)), setStockPrice(sell_history), 'ro')
        plt.show()














