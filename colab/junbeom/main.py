from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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

#################Define Hyperparameters#######################################################################
num_iterations = 100 # @param {type:"integer"}

initial_collect_steps = 1000  # @param {type:"integer"} 
collect_steps_per_iteration = 1  # @param {type:"integer"}
replay_buffer_max_length = 100000  # @param {type:"integer"}

batch_size = 64  # @param {type:"integer"}
learning_rate = 1e-3  # @param {type:"number"}
log_interval = 200  # @param {type:"integer"}

num_eval_episodes = 30  # @param {type:"integer"}
eval_interval = 10  # @param {type:"integer"}
##############################################################################################################
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
#utils.validate_py_environment(env,episodes=10) 

# wrapper py_env -> tf_env
tf_env = tf_py_environment.TFPyEnvironment(env)
#train_env = tf_py_environment.TFPyEnvironment(env)
#eval_env = tf_py_environment.TFPyEnvironment(env)


##################################################################################################################
################################################에이전트##########################################################


# tf_agents.networks.q_network 모듈의 QNetwork 클래스는 
# Q-Learning에 사용되는 인공신경망 (Neural Network)입니다.
#print(q_net) # <tf_agents.networks.q_network.QNetwork object at 0x00000220199A4F40>
#print(q_net.input_tensor_spec) # BoundedTensorSpec(shape=(1,), dtype=tf.int32, name='observation', minimum=array(0), maximum=array(2147483647))
q_net = q_network.QNetwork(
    train_env.observation_spec(), # 모름
    train_env.action_spec(),
    # fc_layer_params = (100,) 
    # creating a network single hidden layer of 100 nodes
    fc_layer_params=(100,) # 신경망의 레이어별 뉴런 유닛의 개수를 지정합니다.
)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
train_step_counter = tf.Variable(0)
#train_step_counter = tf.random.uniform([0],0) # 수정 필요할듯

tf_agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network = q_net, # custom 수정 필요
    optimizer = optimizer, # custom 해도 됨
    # 타겟과 출력값의 오차를 계산하기 위한 함수를 지정합니다.
    td_errors_loss_fn=common.element_wise_squared_loss, #mse 함수
    #train_step_counter=train_step_counter # ??
    # 지정한 tf.varable 은 훈련이 한번 이루어질때마다 값이 1씩 증가
    train_step_counter = tf.Variable(0) #tf.compat.v2.Variable(0)
)

tf_agent.initialize() # 에이전트를 초기화


##################################################################################################################
##################################################정책############################################################


""" Define Policies """
eval_policy = tf_agent.policy #에이전트 현재 정책 반환<tf_agents.policies.greedy_policy.GreedyPolicy object at 0x00000224BD528550>
collect_policy = tf_agent.collect_policy #에이전트가 환경으로부터 데이터를 수집하는 정책 반환 <tf_agents.policies.epsilon_greedy_policy.EpsilonGreedyPolicy object at 0x00000224BD525400>
""" random policies """
# 에이전트와 독립적이며, train_env의 특정 TimeStep에 대해 임의이ㅡ 행동을 반환하는 정책입니다.
random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
    train_env.action_spec())


##################################################################################################################
################################################재생 버퍼############################################################

""" 수정해봐야 할 메소드"""

""" buffer """
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec = tf_agent.collect_data_spec,
    batch_size = train_env.batch_size,
    max_length = replay_buffer_max_length # 하이퍼 파라미터
)
#print(replay_buffer) # <tf_agents.replay_buffers.tf_uniform_replay_buffer.TFUniformReplayBuffer object at 0x0000023B1D53DAC0>

replay_observer = [replay_buffer.add_batch] # ?
#print(replay_observer) # [<bound method ReplayBuffer.add_batch of <tf_agents.replay_buffers.tf_uniform_replay_buffer.TFUniformReplayBuffer object at 0x0000023B1D53DAC0>>]


# 정책 평가 함수
def collect_step(environment,policy,buffer):
    # 주어진 환경에서 주어진 정책을 사용해서 데이터를 수집합니다.
    time_step = environment.current_time_step() #?
    action_step = policy.action(time_step) # PolicyStep(action=<tf.Tensor: shape=(1,), dtype=int32, numpy=array([1])>, state=(), info=())
    #print(f'action : {action_step.action}') # 0,1,2 로 나옴
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step,action_step,next_time_step)
    replay_buffer.add_batch(traj)
    #print(f'collect_step : {time_step}')
    #print("collect_step 값 출력")
    #print(f'time_step {time_step}')
    #print(f'action_step {action_step}') #PolicyStep(action=<tf.Tensor: shape=(1,), dtype=int32, numpy=array([0])>, state=(), info=())
    #print(f'replay_buffer {replay_buffer}') # <tf_agents.replay_buffers.tf_uniform_replay_buffer.TFUniformReplayBuffer object at 0x0000015559F24CD0>
    '''
    time_step -> next_time_step 같은 타입 결과
    TimeStep(
        {'discount': <tf.Tensor: shape=(1,), dtype=float32, numpy=array([0.], dtype=float32)>,
        'observation': <tf.Tensor: shape=(1, 1), dtype=int32, numpy=array([[0]])>,
        'reward': <tf.Tensor: shape=(1,), dtype=float32, numpy=array([0.], dtype=float32)>,
        'step_type': <tf.Tensor: shape=(1,), dtype=int32, numpy=array([2])>})

    Trajectory(
        {'action': <tf.Tensor: shape=(1,), dtype=int32, numpy=array([0])>,
        'discount': <tf.Tensor: shape=(1,), dtype=float32, numpy=array([1.], dtype=float32)>,
        'next_step_type': <tf.Tensor: shape=(1,), dtype=int32, numpy=array([0])>,
        'observation': <tf.Tensor: shape=(1, 1), dtype=int32, numpy=array([[0]])>,
        'policy_info': (),
        'reward': <tf.Tensor: shape=(1,), dtype=float32, numpy=array([0.], dtype=float32)>,
        'step_type': <tf.Tensor: shape=(1,), dtype=int32, numpy=array([2])>})
    '''

""" 수정해봐야 할 메소드"""
def collect_data(env,policy,buffer,steps):
    # 여러 회 (steps)의 데이터 수집 작업을 수행합니다.
    for _ in range(steps):
        collect_step(env,policy,buffer) #??

#initial_collect_steps = 100
######################### 임시 주석
collect_data(train_env,random_policy,replay_buffer,steps=100) # ??? 뭔지 모름

# TFUniformReplayBuffer 객체의 as_dataset() 메서드는 버퍼로부터
# 주어진 형식으로 만들어진 데이터셋을 반환하도록 합니다.
dataset = replay_buffer.as_dataset(
    num_parallel_calls=3,
    sample_batch_size = batch_size,
    num_steps=2
).prefetch(3)


iterator = iter(dataset) # iter() 함수를 사용해서 데이터셋을 반복 가능한 객체로 변환하고,
# .next() 를 사용해서 수집한 데이터를 확인할 수 있습니다.
#print(iterator) # <tensorflow.python.data.ops.iterator_ops.OwnedIterator object at 0x0000023B59D3A580>

##################################################################################################################
##############################################에이전트 교육########################################################

tf_agent.train_step_counter.assign(0)

# 주어진 에피소드 동안의 평균 리턴(보상의 총합의 평균)
#avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes) # 원본
avg_return = compute_avg_return(eval_env, random_policy, num_eval_episodes)
returns = [avg_return]
print(f'avg_return : {avg_return}')
print("h")


for _ in range(num_iterations): # 전체 훈련 횟수

    collect_data(train_env, random_policy, replay_buffer, collect_steps_per_iteration)

    experience, unused_info = next(iterator)
    #print(experience) # 'step_type': <tf.Tensor: shape=(64, 2), dtype=int32, numpy=
    #print(unused_info) # BufferInfo(ids=<tf.Tensor: shape=(64, 2), dtype=int64, numpy= .... dtype=int64)>, probabilities=<tf.Tensor: shape=(64,), dtype=float32, numpy= ......dtype=float32)>)

    train_loss = tf_agent.train(experience).loss

    step = tf_agent.train_step_counter.numpy() # 정수 step 1부터 계속 증가

    if step % log_interval == 0:
        print('step = {0}: loss = {1}'.format(step, train_loss))

    if step % eval_interval == 0: # 훈련 과정 중 검증을 수행할 간격
        print("==================================================================")
        avg_return = compute_avg_return(eval_env, random_policy, num_eval_episodes) # ??
        print('step = {0}: Average Return = {1}'.format(step, avg_return))
        returns.append(avg_return)

print(returns)

import matplotlib.pyplot as plt

iterations = range(0, num_iterations + 1, eval_interval)
plt.xlabel('Iterations')
plt.ylabel('Average Return')
plt.ylim(top=250)
plt.plot(iterations, returns)
plt.show()

print("end")

