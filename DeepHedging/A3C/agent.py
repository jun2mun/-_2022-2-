import gym
from A3C.actor import Actor
from A3C.critic import Critic
from A3C.worker import WorkerAgent
from multiprocessing import cpu_count
import cartpole_env

class Agent:
    def __init__(self, env):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_bound = env.action_space.high[0]
        self.std_bound = [1e-2, 1.0]

        self.global_actor = Actor(
            self.state_dim, self.action_dim, self.action_bound, self.std_bound)
        self.global_critic = Critic(self.state_dim)
        self.num_workers = cpu_count()

    def train(self, max_episodes=1000):
        workers = []

        for i in range(self.num_workers):

            workers.append(WorkerAgent(
                self.env, self.global_actor, self.global_critic, max_episodes))

        for worker in workers:
            worker.start()

        for worker in workers:
            worker.join()