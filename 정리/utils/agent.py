import tensorflow as tf

class Agent(object):
    def __init__(self,time_steps,batch_size, features,\
        nodes = [62,46,46,1], name='model'):
        # 1. 변수 초기화
        tf.compat.v1.reset_default_graph()
        self.batch_size = batch_size # 배치의 옵션 수