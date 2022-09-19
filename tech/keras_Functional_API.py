# 1) 전결합 피드 포워드 신경망(Fully-connected FFNN)
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

inputs = Input(shape=(10,)) # 10개의 입력을 받는 입력층 (입력의 크기를 정의)
hidden1 = Dense(64,activation='relu')(inputs)
hidden2 = Dense(64,activation='relu')(hidden1)
output = Dense(1,activation='sigmoid')(hidden2)
model = Model(inputs=inputs,outputs=output)

model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
# model.fit(data,labels)

inputs = Input(shape=(10,))
x = Dense(8,activation="relu")(inputs)
x = Dense(4,activation="relu")(x)
x = Dense(1,activation="linear")(x)
model = Model(inputs,x)

# 2) 선형 회귀

# 5) RNN(Recurrence Neural Network) 은닉층 사용하기 #
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.models import Model

inputs = Input(shape=(50,1))
lstm_layer = LSTM(10)(inputs)
x = Dense(10,activation='relu')(lstm_layer)
output = Dense(1,activation='sigmoid')(x)

model = Model(inputs=inputs,outputs=output)