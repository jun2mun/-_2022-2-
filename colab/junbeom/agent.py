import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

from environment import TradeEnv

class A2CAgent():
    def __init__(self,state_size,action_size):
        self.load_model = False
        self.state_size = state_size
        self.action_size = action_size
        self.value_size = 1 #???

        # actor - critic hyperparameter
        self.discount_factor = 0.99 # ??
        self.actor_lr = 0.001 # ??
        self.critic_lr = 0.005

        # 정책신경망과 가치신경망 생성
        self.actor = self.build_actor()
        self.critic = self.build_critic()
        #self.actor_updater = self.actor_optimizer()
        #self.critic_updater = self.critic_optimizer()

        if self.load_model:
            self.actor.load_weights("./save_model/cartpole_actor_trained.h5")
            self.critic.load_weights("./save_model/cartpole_critic_trained.h5")    

    def build_actor(self):
        actor = Sequential()
        actor.add(Dense(24, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        actor.add(Dense(self.action_size, activation='softmax',
                        kernel_initializer='he_uniform'))
        actor.summary()
        return actor

        # critic: 상태를 받아서 상태의 가치를 계산
    def build_critic(self):
        critic = Sequential()
        critic.add(Dense(24, input_dim=self.state_size, activation='relu',
                         kernel_initializer='he_uniform'))
        critic.add(Dense(24, input_dim=self.state_size, activation='relu',
                         kernel_initializer='he_uniform'))
        critic.add(Dense(self.value_size, activation='linear',
                         kernel_initializer='he_uniform'))
        critic.summary()
        return critic

    # 정책신경망의 출력을 받아 확률적으로 행동을 선택
    def get_action(self, state):
        policy = self.actor.predict(state, batch_size=1).flatten()
        return np.random.choice(self.action_size, 1, p=policy)[0]
        

    pass

def custom_agent(S,balance):

    env = TradeEnv(S,balance)
    eval_env = TradeEnv(S,balance)

    state_size = env._observation_spec.shape[0]
    action_size = env._action_spec.maximum - env._action_spec.minimum + 1
    
    agent = A2CAgent(state_size,action_size)

    scores, episodes =[] , []

    EPISODES = 100
    for ep in range(EPISODES):
        done = False
        score = 0
        state = env.reset()

        while not done:
            action = agent.get_action(state)
            print(action)
            #next_state, reward, done, info = env.step(action)
            #next_state = np.reshape(next_state,[1,state_size])
