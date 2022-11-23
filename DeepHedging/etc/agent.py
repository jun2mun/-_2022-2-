import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.utils import common

optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
global_step = tf.compat.v1.train.get_or_create_global_step()
train_step_counter = tf.Variable(0)

def getAgent(env, q_network):
    tf_agent = dqn_agent.DqnAgent(
        env.time_step_spec(),
        env.action_spec(),
        q_network=q_network,  # custom 수정 필요
        optimizer=optimizer,  # custom 해도 됨
        # 타겟과 출력값의 오차를 계산하기 위한 함수를 지정합니다.
        td_errors_loss_fn=common.element_wise_squared_loss,  # mse 함수
        # 지정한 tf.varable 은 훈련이 한번 이루어질때마다 값이 1씩 증가
        # train_step_counter = global_step #tf.compat.v2.Variable(0)
        train_step_counter=train_step_counter
    )

    return tf_agent