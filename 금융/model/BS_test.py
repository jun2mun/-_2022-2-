import tensorflow as tf
import matplotlib.pyplot as plt

def BS_test(train_dataset,y,shape):
    model = tf.keras.models.Sequential()

    ### RNN 파트 시작점 ###

    # 단일 SimpleRNN, 단일 방향 예시
    model.add(tf.keras.layers.SimpleRNN(16)) # 유닛 개수 = 16 예시(변경 가능)

    ### RNN 파트 끝점 ###

    # fc layer 부분(32 차원 변환 -> dropout -> 이진 분류 결과)
    model.add(tf.keras.layers.Dense(32, activation = 'relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(1, activation = 'sigmoid')) # 이진 분류를 위한 마지막 layer 설정

    # 선언 모델 학습 부분
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

    hist = model.fit(train_dataset,y, batch_size=32, epochs=1000,  verbose=True, validation_split=0.2)
    model.save('BS_TEST_name.h5')

    plt.plot(hist.history['loss'])
    plt.ylabel('loss')
    plt.xlabel('iter')
    plt.show()