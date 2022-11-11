from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from tf_agents.policies import random_tf_policy, PolicySaver

from environment import TradeEnv
from utils import monte_carlo_paths, compute_avg_return

from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import tf_py_environment
from tf_agents.utils import common
from tf_agents.trajectories import trajectory
from tf_agents.networks import q_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
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

    # tf_agents.networks.q_network 모듈의 QNetwork 클래스는
    # Q-Learning에 사용되는 인공신경망 (Neural Network)입니다.
    #q_net = q_network.QNetwork(
    #    train_env.observation_spec(),
    #    train_env.action_spec(),
    #    fc_layer_params=(100,) # 신경망의 레이어별 뉴런 유닛의 개수를 지정합니다.
    #)
    from tf_agents.specs import tensor_spec
    fc_layer_params = (100, 50)
    action_tensor_spec = tensor_spec.from_spec(env.action_spec())
    num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1

    # Define a helper function to create Dense layers configured with the right
    # activation and kernel initializer.
    def dense_layer(num_units):
        return tf.keras.layers.Dense(
        num_units,
        activation=tf.keras.activations.relu,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=2.0, mode='fan_in', distribution='truncated_normal'))

    # QNetwork consists of a sequence of Dense layers followed by a dense layer
    # with `num_actions` units to generate one q_value per available action as
    # its output.
    from tf_agents.networks import sequential
    dense_layers = [dense_layer(num_units) for num_units in fc_layer_params]
    q_values_layer = tf.keras.layers.Dense(
        num_actions,
        activation=None,
        kernel_initializer=tf.keras.initializers.RandomUniform(
            minval=-0.03, maxval=0.03),
        bias_initializer=tf.keras.initializers.Constant(-0.2))
    q_net = sequential.Sequential(dense_layers + [q_values_layer])

    """=========================에이전트 구성하기============================"""

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
    global_step = tf.compat.v1.train.get_or_create_global_step()
    train_step_counter = tf.Variable(0)

    tf_agent = dqn_agent.DqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network = q_net, # custom 수정 필요
        optimizer = optimizer, # custom 해도 됨
        # 타겟과 출력값의 오차를 계산하기 위한 함수를 지정합니다.
        td_errors_loss_fn=common.element_wise_squared_loss, #mse 함수
        # 지정한 tf.varable 은 훈련이 한번 이루어질때마다 값이 1씩 증가
        #train_step_counter = global_step #tf.compat.v2.Variable(0)
        train_step_counter=train_step_counter
    )

    tf_agent.initialize() # 에이전트를 초기화

    """=======================에이전트의 정책============================="""

    """ Define Policies """
    eval_policy = tf_agent.policy #에이전트 현재 정책 반환<tf_agents.policies.greedy_policy.GreedyPolicy object at 0x00000224BD528550>
    collect_policy = tf_agent.collect_policy #에이전트가 환경으로부터 데이터를 수집하는 정책 반환
    random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
        train_env.action_spec()) # 에이전트와 독립적이며, train_env의 특정 TimeStep에 대해 임의의 행동을 반환하는 정책입니다.

    """ buffer """
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec = tf_agent.collect_data_spec,
        batch_size = train_env.batch_size,
        max_length = replay_buffer_max_length # 하이퍼 파라미터
    )
    replay_observer = [replay_buffer.add_batch]
    '''
    checkpoint_dir = os.path.join("./checkpoint", "checkpoint_{}".format(1))
    train_checkpointer = common.Checkpointer(
        ckpt_dir=checkpoint_dir,
        max_to_keep=1,
        agent=tf_agent,
        policy=tf_agent.policy,
        replay_buffer=replay_buffer,
        global_step=global_step
    )
    train_checkpointer.initialize_or_restore()
    '''

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