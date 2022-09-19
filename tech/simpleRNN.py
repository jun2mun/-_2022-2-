from json import encoder
import numpy as np
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import layers
from tensorflow.python.keras.layers import SimpleRNN
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Embedding
from tensorflow.python.keras import Sequential
## SimpleRNN 레이어 ##
#rnn1 = SimpleRNN(units=1,activation='tanh',return_sequences=True)
'''
# input_shape [4,1] timesteps, input_dim
# timesteps 순환 신경망이 입력에 대해 계산을 반복하는 횟수
# input_dim 벡터의 크기
model = Sequential([
    SimpleRNN(units=10,return_sequences=False,input_shape=[4,1]),
    layers.Dense(1)
])

model.compile(optimizer='adam',loss='mse')
model.summary()


model = Sequential()
model.add(layers.Embedding(input_dim=1000,output_dim=64))
model.add(layers.LSTM(128))
model.add(layers.Dense(10))
model.summary()

model = Sequential()
model.add(layers.Embedding(input_dim=1000,output_dim=64))
model.add(layers.GRU(256,return_sequences=True))
model.add(layers.SimpleRNN(128))
model.add(layers.Dense(10))
model.summary()
'''

encoder_vocab = 1000
decoder_vocab = 2000
encoder_input = layers.Input(shape=(None,))
encoder_embedded = layers.Embedding(input_dim=encoder_vocab,output_dim=64)(
    encoder_input
)

output, state_h, state_c = layers.LSTM(64,return_state=True,name="encoder")(
    encoder_embedded
)
encoder_state = [state_h,state_c]

decoder_input = layers.Input(shape=(None,))
decoder_embedded = layers.Embedding(input_dim=decoder_vocab,output_dim=64)(
    decoder_input
)

decoder_output = layers.LSTM(64, name="decoder")(
    decoder_embedded, initial_state=encoder_state
)
output = layers.Dense(10)(decoder_output)

model = keras.Model([encoder_input,decoder_input],output)
model.summary()