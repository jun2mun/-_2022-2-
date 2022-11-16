from A3C.agent import Agent
from cartpole_env import CartPoleEnv
from pendulum_env import PendulumEnv
# env = CartPoleEnv()
env = PendulumEnv()
agent = Agent(env)
agent.train()