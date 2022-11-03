from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


""" 1번 과제 """
#import tf_agents

#print(tf_agents.version.__version__) # 0.14.0
#print(tf_agents.version._MAJOR_VERSION)
#print(tf_agents.version._MINOR_VERSION)
#print(tf_agents.version._PATCH_VERSION)


""" 2번 과제 """
'''
import tf_agents
from tf_agents.environments import suite_gym

env_name = 'CartPole-v0'
env = suite_gym.load(env_name)
print(env)

env.reset()
'''

""" 3번 과제"""
# TimeStep 객체는 step_type, reward, discount, observation
# TF-Agents 환경 (env)의 time_step_spec() 메서드는 TimeStep의 사양을 반환합니다.
# ArraySpec 객체는 NumPy Array, Scalar의 형태 (shape)와 자료형 (dtype)을 표시하기 위해 사용됩니다.
# BoundedArraySpec는 최소, 최대값 범위를 갖는 ArraySpec입니다.

# 강화학습의 환경 (Environment)과 에이전트 (Agent)는 행동 (Action)과 관찰 (Observation)을 통해 상호작용합니다.
#action_spec()을 사용해서 이러한 행동 (Action)의 사양을 확인할 수 있습니다.

# env.reset() 메서드가 환경을 재설정 (reset)하고, 첫 TimeStep 객체를 반환했다면,

# env.step(action)은 환경에 action을 반영한 다음, TimeStep을 반환합니다.

"""  wrapper 란  """

# Python 기반의 환경을 TensorFlow 환경으로 포장 (Wrap)하면, 
# 배열 (Array) 대신 텐서 (Tensor)를 사용하고 
# TensorFlow가 작업을 병렬화할 수 있도록 합니다.

# ArraySpec -> TensorSpec
# BoundedArraySpec -> BoundedTensorSpec

# train_env -> train_py_env
# eval_env -> eval_py_env

""" QNetwork """
# tf_agents.networks.q_network 모듈의 QNetwork 클래스는
# Q-Learning에 사용되는 인공신경망 (Neural Network)입니다.

""" 정책 """
# dqn_agent.DqnAgent 인스턴스는 policy, collect_policy 속성을 포함합니다.
# policy는 에이전트의 현재 정책을, collect_policy는 에이전트가 환경으로부터 데이터를 수집하는 정책을 반환합니다.
# 에이전트와 독립적인 정책을 만들 수 있습니다.
# random_tf_policy 모듈의 RandomTFPolicy 클래스는 주어진 action_spec에서 임의로 샘플링한 행동을 반환합니다.

""" 정책으로부터 행동 얻기"""


""" 정책 평가 함수 만들기 """
# random_tf_policy 모듈의 RandomTFPolicy 클래스는 
# 주어진 action_spec에서 임의로 샘플링한 행동을 반환합니다.

# total_return은 주어진 에피소드 동안 얻은 리턴의 총합이며, 
# episode_return은 각 에피소드의 리턴 (보상의 총합)입니다.

# TimeStep 인스턴스는 is_first(), is_mid(), is_last() 메서드를 포함합니다.
# TimeStep 인스턴스의 is_last() 메서드는 해당 
# TimeStep이 에피소드의 마지막 TimeStep인지 여부를 반환합니다.



import os
import tensorflow as tf
import abc
import tensorflow as tf
import numpy as np

import base64
import matplotlib.pyplot as plt
from tensorflow.python.eager.monitoring import time

from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer


from utils import monte_carlo_paths,compute_avg_return
from environment import TradeEnv

from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import py_environment, utils
from tf_agents.environments import tf_py_environment,tf_environment
from tf_agents.utils import common
from tf_agents.specs import array_spec
from tf_agents.trajectories import trajectory
from tf_agents.trajectories import time_step as ts
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.specs import tensor_spec
from tf_agents.drivers import py_driver
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import policy_saver



S_0 = 100
K = 100
r = 0
vol = 0.2
T = 1 / 12
timesteps = 30
seed = 42
n_sims = 10

# Train the model on the path of the risk neutral measure
paths_train = monte_carlo_paths(S_0, T, vol, r, seed, n_sims, timesteps)

sell_stock = np.array(0, dtype=np.int32) # 0
keep_stock = np.array(1, dtype=np.int32) # 1
buy_new_stock = np.array(2, dtype=np.int32) # 2
strategy = [sell_stock, keep_stock, buy_new_stock]

S = np.swapaxes(paths_train,0,1)[0]  # S.shape = (31,1)
T = 30
balance = 100000
##################################################################################################################
###################################################환경###########################################################

env = TradeEnv(S,T,balance)


tempdir = './colab/junbeom/'
policy_dir = os.path.join(tempdir, 'policy')
saved_policy = tf.saved_model.load(policy_dir)
time_step = env.reset() # timestamp 형식 반환
episode_return = 0.0

tf_env = tf_py_environment.TFPyEnvironment(env)
train_env = tf_py_environment.TFPyEnvironment(env)
eval_env = tf_py_environment.TFPyEnvironment(env)

time_step = eval_env.reset()
action_step = saved_policy.action(time_step)
next_time_step = eval_env.step(action_step.action)

traj = trajectory.from_transition(time_step,action_step,next_time_step)
print(traj)

