from pandas_datareader import data as wb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, LSTM, SimpleRNN, Dropout, Flatten
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from keras.optimizers import SGD
import math


class Finance_Neural_Nets:

    data_column =''
    window = 0
    batch_size = 0
    epoch = 0

    def __init__(self, asset_name, source, start_date, end_date):
        self.asset_name = asset_name
        self.source = source
        self.start_date = start_date
        self.end_date = end_date

    def plot_finance_data(self):
        asset = wb.DataReader(self.asset_name, data_source=self.source, start=self.start_date, end=self.end_date)

        plt.figure(figsize=(16, 8))
        plt.title('Close Price History')
        plt.plot(asset['Close'])
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Close Price', fontsize=18)
        plt.show()

    def prepare_data(self):
        asset = wb.DataReader(self.asset_name, data_source=self.source, start=self.start_date, end=self.end_date)

        data = asset.filter([self.data_column])
        # Convert the dataframe to a numpy array
        dataset = data.values

        # Get the number of rows to train the model on
        training_data_len = math.ceil(len(dataset) * .8)

        # Scale the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)

        train_data = scaled_data[0:training_data_len, :]

        x_train = []
        y_train = []

        for i in range(self.window, len(train_data)):
            x_train.append(train_data[i - self.window:i, 0])
            y_train.append(train_data[i, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)
        print(x_train.shape, 'X train data shape')

        # Reshape the data
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        return x_train, y_train, x_train.shape, training_data_len, scaler, scaled_data, data, dataset

    def neural_net_RNN_model(self, x_train):
        # Build the RNN model
        model = Sequential()
        model.add(SimpleRNN(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(keras.layers.Dropout(0.2))

        model.add(SimpleRNN(50, return_sequences=True))
        model.add(keras.layers.Dropout(0.5))

        model.add(SimpleRNN(50))

        model.add(Dense(1, activation='linear'))
        #print(model.summary())

        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')
        # model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        return model

    def neural_net_LSTM_model(self, x_train):
        # Build the LSTM model
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(keras.layers.Dropout(0.2))

        model.add(LSTM(50, return_sequences=False))
        model.add(keras.layers.Dropout(0.5))

        model.add(Dense(25))
        # model.add(keras.layers.Dropout(0.5))

        model.add(Dense(1, activation='linear'))

        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')
        # model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        return model


    def train_neural_net(self, model, x_train, y_train, training_data_len, scaler, scaled_data, data, dataset):
        # Train the model
        model_hist = model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.epoch, verbose=1)
        print(model.summary(), 'Model summery')

        # Create the testing data set
        # Create a new array containing scaled values from index 1543 to 2002
        test_data = scaled_data[training_data_len - self.window:, :]

        x_test = []
        y_test = dataset[training_data_len:, :]
        for i in range(self.window, len(test_data)):
            x_test.append(test_data[i - self.window:i, 0])

        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)

        # Get the root mean squared error (RMSE)
        rms = np.sqrt(np.mean(np.power((predictions - y_test), 2)))
        rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
        print(rmse, 'RMSE')
        print(rms, 'RMS')

        # Plot the loss
        tra_loss = model_hist.history['loss']
        plt.plot(tra_loss)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curve')
        plt.legend(['Training Loss'])

        # Plot the data
        train = data[:training_data_len]
        valid = data[training_data_len:]
        valid['Predictions'] = predictions
        # Visualize the data
        plt.figure(figsize=(16, 8))
        plt.title('Model')
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Close Price', fontsize=18)
        plt.plot(train['Adj Close'])
        plt.plot(valid[['Adj Close', 'Predictions']])
        plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
        plt.show()

        print(valid.head(), ' valid dataset')

    def save_neural_net_model(self, model, model_name):
        model.save(model_name + 'h5')

    def load_neural_net_model(self, model_name):
        model = load_model(model_name + 'h5')
        return model

    def test_neural_net_model(self, model, scaler):

        apple_quote = wb.DataReader('AAPL', data_source='yahoo', start='2012-01-01', end='2019-12-17')
        new_df = apple_quote.filter([str(self.data_column)])

        last_window_days = new_df[-self.window:].values
        last_window_days_scaled = scaler.transform(last_window_days)

        X_test = []
        X_test.append(last_window_days_scaled)
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        pred_price = model.predict(X_test)

        pred_price = scaler.inverse_transform(pred_price)
        print(pred_price)

        return pred_price

