# 1. 전처리 (Preprocessing) #
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer()
train_text = "The earth is an awesome place live"

    # 단어 집합 생성 #
tokenizer.fit_on_texts([train_text]) ## 동작 원리?>

    # 정수 인코딩 #
sub_text = "The earth is an great place live"
sequences = tokenizer.texts_to_sequences([sub_text])[0] ## 동작 원리?>

print("정수 인코딩 : ",sequences)
print("단어 집합 : ",tokenizer.word_index)

## 패딩 작업 : 입력 글자마다 길이 다를 수 있어서 ##
# 첫번째 인자 = 패딩을 진행할 데이터
# maxlen = 모든 데이터에 대해서 정규화 할 길이
# padding = 'pre'를 선택하면 앞에 0을 채우고 'post'를 선택하면 뒤에 0을 채움
print(pad_sequences([[1,2,3],[3,4,5,6],[7,8]],maxlen=3,padding='pre'))


# 2. 워드 임베딩 ( Word Embedding) #
from tensorflow.keras.layers import Embedding
# Embedding() => (number of samples,input_length) 입력 받는다
# return => (number of samples,input_length,embedding word dimensionality)

    #1. 토큰화
tokenized_text = [['Hope', 'to', 'see', 'you', 'soon'], ['Nice', 'to', 'see', 'you', 'again']]
    #2. 각 단어에 대한 정수 인코딩
encoded_text = [[0, 1, 2, 3, 4],[5, 1, 2, 3, 6]]
    #3. 위 정수 인코딩 데이터가 아래의 임베딩 층의 입력이 된다.
vocab_size = 7 # 0~6 7개
embedding_dim = 2 # 차원은 2개

# 첫 번째 인자 : vocab_size개의 단어 종류
# 두 번째 인자 : 출력의 차원 
Embedding(vocab_size,embedding_dim,input_length=5)

# 각 정수는 아래의 테이블릐 인덱스로 사용되며 Embedding()은 각 단어마다 임베딩 벡터를 리턴한다.

# 3. 모델링 (Modeling) #
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(8, input_dim=4,activation='relu'))
model.add(Dense(1, activation='sigmoid')) # 출력층
model.summary()

# 4. 컴파일(Compile) #
from tensorflow.keras.layers import SimpleRNN, Embedding, Dense
from tensorflow.keras.models import Sequential

vocab_size = 10000
embedding_dim = 32
hidden_units = 32

model = Sequential()
model.add(Embedding(vocab_size,embedding_dim))
model.add(SimpleRNN(hidden_units))
model.add(Dense(1,activation='sigmoid'))
# optimizer
# loss
# metrics = 훈련을 모니터링하기 위한 지표를 선택합니다.
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])

# 5.  훈련(Training) #
# 첫번째
# 두번째
# epochs
# batch_size
# validation_data(x_val,y_val)
# validation_split
# verbose : 0(아무것도 출력 X) 1(훈련의 진행도를 부여주는 진행 막대를 보여줍니다.) 2(미니배치마다 손실정보를 출력합니다.)
#model.fit(X_train,y_train,epochs=10,batch_size=32)
#model.fit(X_train,y_train,epochs=10,batch_size=32,verbose=0,validation_data(X_val,y_val))
#model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0, validation_split=0.2))

# 6. 평가(Evaluation)와 예측(Prediction)
#테스트 데이터를 통해 학습한 모델에 대한 정확도를 평가합니다.
# 첫번째
# 두번째
# batch_size
#model.evaluate(X_test,y_test,batch_size=32)

# 첫번째 = 예측하고자 하는 데이터
# batch_size
#model.predict(X_input,batch_size=32)

# 7. 모델의 저장(Save) 과 로드(Load) #
#model.save('model_name.h5')
#from tensorflow.keras.models import load_model
#model = load_model("model_name.h5")