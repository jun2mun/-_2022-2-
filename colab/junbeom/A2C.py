from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from tf_agents.policies import random_tf_policy, PolicySaver

from environment import TradeEnv
from utils import monte_carlo_paths, compute_avg_return

from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.sac import sac_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_pybullet
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import normal_projection_network
from tf_agents.policies import greedy_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common


import matplotlib.pyplot as plt

#Define Hyperparameters
num_iterations = 200 # @param {type:"integer"} 훈련 횟수
collect_steps_per_iteration = 30  # @param {type:"integer"} 훈련 당 데이터 수집 횟수
replay_buffer_max_length = 100000  # @param {type:"integer"} 버퍼 최대 크기

batch_size = 64  # @param {type:"integer"} 배치 사이즈
learning_rate = 1e-3  # @param {type:"number"}
log_interval = 10  # @param {type:"integer"} 훈련 로그 출력 간격

num_eval_episodes = 10  # @param {type:"integer"} 검증 데이터 수집 횟수
eval_interval = 5  # @param {type:"integer"} 검증 간격

T = 31 # @param {type:"integer"} 만기일
balance = 10000 # @param {type:"integer"} 초기 자본금
"""================================================================"""

def train(S,S2):


    """=========================Environment 구성============================"""

    env = TradeEnv(S,balance)
    eval_env = TradeEnv(S,balance)

    # wrapper py_env -> tf_env
    train_env = tf_py_environment.TFPyEnvironment(env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_env)

    """=========================Q network 구성============================"""

    observation_spec = train_env.observation_spec()
    action_spec = train_env.action_spec()
    critic_net = critic_network.CriticNetwork(
        (observation_spec, action_spec),
          observation_fc_layer_params=None,
          action_fc_layer_params=None,
          joint_fc_layer_params=critic_joint_fc_layer_params

    def normal_projection_net(action_spec,init_means_output_factor=0.1):
      return normal_projection_network.NormalProjectionNetwork(
          action_spec,
          mean_transform=None,
          state_dependent_std=True,
          init_means_output_factor=init_means_output_factor,
          std_transform=sac_agent.std_clip_transform,
          scale_distribution=True)
    actor_net = actor_distribution_network.ActorDistributionNetwork(
        observation_spec,
        action_spec,
        fc_layer_params=actor_fc_layer_params,
        continuous_projection_net=normal_projection_net)
    

    global_step = tf.compat.v1.train.get_or_create_global_step()
    tf_agent = sac_agent.SacAgent(
        train_env.time_step_spec(),
        action_spec,
        actor_network=actor_net,
        critic_network=critic_net,
        actor_optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=actor_learning_rate),
        critic_optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=critic_learning_rate),
        alpha_optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=alpha_learning_rate),
        target_update_tau=target_update_tau,
        target_update_period=target_update_period,
        td_errors_loss_fn=tf.compat.v1.losses.mean_squared_error,
        gamma=gamma,
        reward_scale_factor=reward_scale_factor,
        gradient_clipping=gradient_clipping,
        train_step_counter=global_step)
    tf_agent.initialize()

    # 정책 평가 함수
    def collect_step(environment,policy,buffer):
        # 주어진 환경에서 주어진 정책을 사용해서 데이터를 수집합니다.
        time_step = environment.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = environment.step(action_step.action)
        traj = trajectory.from_transition(time_step,action_step,next_time_step)
        replay_buffer.add_batch(traj)


    # 데이터 수집 함수
    def collect_data(env,policy,buffer,steps):
        env.reset()
        for _ in range(steps): # 여러 회 (steps)의 데이터 수집 작업을 수행합니다.
            collect_step(env,policy,buffer)

    """=======================replay buffer 생성============================="""

    collect_data(train_env,random_policy,replay_buffer,steps=31) #랜덤 정책으로 데이터 수집

    # 주어진 형식으로 만들어진 데이터셋을 반환하도록 합니다.
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size = batch_size,
        num_steps=2
    ).prefetch(3)

    iterator = iter(dataset) # iter() 함수를 사용해서 데이터셋을 반복 가능한 객체로 변환하고,

    # Training Agent
    tf_agent.train = common.function(tf_agent.train)
    tf_agent.train_step_counter.assign(0)
    """=======================train 시작============================="""
    # 주어진 에피소드 동안의 평균 리턴(보상의 총합의 평균)
    avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
    returns = [avg_return]
    print(f'avg_return : {avg_return}')


    for idx in range(num_iterations): # 전체 훈련 횟수

        collect_data(train_env, tf_agent.collect_policy, replay_buffer, collect_steps_per_iteration)

        experience, unused_info = next(iterator)
        train_loss = tf_agent.train(experience).loss

        step = tf_agent.train_step_counter.numpy() # 정수 step 1부터 계속 증가

        if step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss))

        if step % eval_interval == 0: # 훈련 과정 중 검증을 수행할 간격
            print("=========================== 검증 시작 =================================")
            avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
            print('step = {0}: Average Return = {1}'.format(step, avg_return))
            returns.append(avg_return)


    #train_checkpointer.save(global_step) #model 저장
    print(returns)



    iterations = range(0, num_iterations + 1, eval_interval)
    plt.xlabel('Iterations')
    plt.ylabel('Average Return')
    #plt.ylim([min(returns) * 0.99, max(returns) * 1.01])
    plt.plot(iterations, returns)
    plt.show()

    print('jhi')

