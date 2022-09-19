from tensorflow.keras.layers import SimpleRNN

'''
# basic
model.add(SimpleRNN(hidden_units))
# 추가 인자를 사용할때
model.add(SimpleRNN(hidden_units,input_shape=(timesteps,input_dim)))
# 다른 표기
model.add(SimpleRNN(hidden_units,input_length=M,input_dim=N))
'''

from tensorflow.keras.models import Sequential

model = Sequential()
model.add(SimpleRNN(3, input_shape=(2,10))) # 3은 hidden_state
# model.add(SimpleRNN(3,input_length=2,input_dim=10))
model.summary() # (None,3) -> batch_size를 현 단계에서는 알 수 없으므로

model = Sequential()
model.add(SimpleRNN(3, batch_input_shape=(8,2,10)))
model.summary() # batch_size가 8이므로 (8,3)

model = Sequential()
# (batch_size,timesteps,output_dim) = True
# (batch_size,output_dim) = False(default)
model.add(SimpleRNN(3, batch_input_shape=(8,2,10), return_sequences=True))
model.summary()

