import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.metrics import MeanSquaredError
import matplotlib.pyplot as plt


def prediction(data, commodity):
    # Select a single commodity for prediction, for example, 'Tomato Big(Nepali)'
    df = data[data["Commodity"] == commodity]["Average"].values.reshape(-1, 1)

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scaler.fit_transform(df)

    # Split into training and testing data
    split = int(0.8 * len(df_scaled))
    train = df_scaled[:split]
    test = df_scaled[split:]

    # Convert an array of values into a dataset matrix

    def create_dataset(dataset, look_back=10):
        X, Y = [], []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i : (i + look_back), 0]
            X.append(a)
            Y.append(dataset[i + look_back, 0])
        return np.array(X), np.array(Y)

    # Reshape into X=t and Y=t+1
    look_back = 10
    X_train, Y_train = create_dataset(train, look_back)
    X_test, Y_test = create_dataset(test, look_back)

    # Reshape input to be [samples, time steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    # Build the LSTM network
    model = Sequential()
    model.add(LSTM(50, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error", optimizer="adam")

    # Fit the model
    model.fit(X_train, Y_train, epochs=10, batch_size=1, verbose=2)

    # Make predictions
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    # Invert predictions to return to original scale
    train_predict = scaler.inverse_transform(train_predict)
    Y_train = scaler.inverse_transform([Y_train])
    test_predict = scaler.inverse_transform(test_predict)
    Y_test = scaler.inverse_transform([Y_test])

    # Calculate root mean squared error
    train_score = np.sqrt(MeanSquaredError()(Y_train[0], train_predict[:, 0]))
    test_score = np.sqrt(MeanSquaredError()(Y_test[0], test_predict[:, 0]))
    print("Train Score: %.2f RMSE" % (train_score))
    print("Test Score: %.2f RMSE" % (test_score))

    return test_predict[:, 0]
