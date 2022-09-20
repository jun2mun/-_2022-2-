def LSTM_model(X_train,y_train,X_test, sc):
    from keras.models import Sequential
    from keras.layers import Dense, SimpleRNN, GRU, LSTM
    from tensorflow.keras.optimizers import SGD #https://stackoverflow.com/questions/67604780/unable-to-import-sgd-and-adam-from-keras-optimizers


    my_LSTM_model = Sequential()
    my_LSTM_model.add(LSTM(units=50,
        return_sequences=True,
        input_shape = (X_train.shape[1],1),
        activation='tanh'))
    my_LSTM_model.add(LSTM(units=50, activation='tanh'))
    my_LSTM_model.add(Dense(units=2))

    my_LSTM_model.compile(optimizer = SGD(learning_rate= 0.01, decay = 1e-7,
        momentum = 0.9, nesterov = False),
        loss='mean_squared_error')

    my_LSTM_model.fit(X_train,y_train,epochs=50,batch_size =150, verbose = 0)
    LSTM_predcition = my_LSTM_model.predict(X_test)
    LSTM_prediction = sc.inverse_transform(LSTM_predcition)

    return my_LSTM_model, LSTM_prediction