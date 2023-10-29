# File: stock_prediction.py
# Authors: Cheong Koo and Bao Vo
# Date: 14/07/2021(v1); 19/07/2021 (v2); 25/07/2023 (v3)

# Code modified from:
# Title: Predicting Stock Prices with Python
# Youtuble link: https://www.youtube.com/watch?v=PuZY9q-aKLw
# By: NeuralNine

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as datetime
import tensorflow as tf
import yfinance as yf
import os
import mplfinance as fplt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, InputLayer, GRU, SimpleRNN, Bidirectional
from sklearn.model_selection import train_test_split

DATA_SOURCE = "yahoo"
COMPANY = "TSLA"
DATA_FILE = "TSLA_data.csv"
TRAIN_START = '2015-01-01'
TRAIN_END = '2020-01-01'
TEST_START = '2020-01-02'
TEST_END = '2022-12-31'
PRICE_VALUE = "Close"
PREDICTION_DAYS = 60
sequence_length = 60 
features = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
n_features = 6


def start_end_date (TRAIN_START_STR ='2025-01-01', TRAIN_END_STR = '2020-01-01', date_format = "%Y-%m-%d"):
    train_start = datetime.strptime(TRAIN_START_STR, date_format)
    train_end = datetime.strptime(TRAIN_END_STR, date_format)
    print(train_start,train_end)
    if train_start < train_end:
        print("Error: Train start date must be earlier than train end date!")
    return train_start, train_end

def handle_missing_values(df):
    if df.isnull().sum().sum() > 0:
        df.fillna(method='ffill', inplace=True)
    return df

def custom_train_test_split(data, split_ratio=0.8, method='random'):
    
    train_data, test_data = train_test_split(data, train_size=split_ratio, random_state=42)
    
    return train_data, test_data

def data_to_file(DATA_FILE):
    if os.path.exists(DATA_FILE): 
        print(f"Loading data from {DATA_FILE}")
        data = pd.read_csv(DATA_FILE)
    else:
        print(f"Fetching data from {DATA_SOURCE} for {COMPANY}")
        data = yf.download(COMPANY, start=TRAIN_START, end=TRAIN_END, progress=False)
        print(f"Saving data to {DATA_FILE}")
        data.to_csv(DATA_FILE)
    return 

def prepare_multivariate_multistep_data(data, features, steps_in, steps_out):
    X, y = [], []
    data_df = pd.DataFrame(data, columns=features)
    for i in range(len(data_df) - steps_in - steps_out + 1):
        seq_in = data_df.iloc[i:i + steps_in][features].values
        seq_out = data_df.iloc[i + steps_in:i + steps_in + steps_out]["Close"].values
        X.append(seq_in)
        y.append(seq_out)
    return np.array(X), np.array(y)


def custom_model(sequence_length, n_features, layer_type=LSTM, n_layers=2, layer_units=50, dropout=0.3, loss="mean_absolute_error", optimizer="rmsprop", bidirectional=False):
  
    model = Sequential()

    for i in range(n_layers):
        if i == 0:
            # First layer
            if bidirectional:
                model.add(Bidirectional(layer_type(layer_units, return_sequences=True), input_shape=(sequence_length, n_features)))
            else:
                model.add(layer_type(layer_units, return_sequences=True, input_shape=(sequence_length, n_features)))
        elif i == n_layers - 1:
            # Last layer
            if bidirectional:
                model.add(Bidirectional(layer_type(layer_units, return_sequences=False)))
            else:
                model.add(layer_type(layer_units, return_sequences=False))
        else:
            # Hidden layers
            if bidirectional:
                model.add(Bidirectional(layer_type(layer_units, return_sequences=True)))
            else:
                model.add(layer_type(layer_units, return_sequences=True))
        
        # Add dropout after each layer
        model.add(Dropout(dropout))

    # Output layer
    model.add(Dense(steps_out, activation="linear"))

    # Compile the model
    model.compile(loss=loss, metrics=["mean_absolute_error"], optimizer=optimizer)

    return model

data =  yf.download(COMPANY, start=TRAIN_START, end=TRAIN_END, progress=False)

data = handle_missing_values(data)

scaler = MinMaxScaler(feature_range=(0, 1)) 

full_data = yf.download(COMPANY, start=TRAIN_START, end=TEST_END, progress=False)
full_data = handle_missing_values(full_data)
scaled_full_data = scaler.fit_transform(full_data[features].values)

train_data_length = int(len(scaled_full_data) * 0.8)
train_data = scaled_full_data[:train_data_length]
test_data = scaled_full_data[train_data_length - PREDICTION_DAYS:]

scaled_data = scaler.fit_transform(data[features].values.reshape(-1, 1)) 

steps_in = PREDICTION_DAYS
steps_out = 5

X_train_multivariate_multistep, y_train_multivariate_multistep = prepare_multivariate_multistep_data(train_data, features, steps_in, steps_out)
X_test_multivariate_multistep, y_test_multivariate_multistep = prepare_multivariate_multistep_data(test_data, features, steps_in, steps_out)

lstm_model = custom_model(sequence_length, n_features, layer_type=LSTM, n_layers=2, 
                                 layer_units=100, dropout=0.2, optimizer="adam")


lstm_model.fit(X_train_multivariate_multistep, y_train_multivariate_multistep, epochs=5, batch_size=32)
lstm_model.compile(optimizer="adam", loss="mean_absolute_error", metrics=["mean_absolute_error"], run_eagerly=True)

predicted_closing_prices = lstm_model.predict(X_test_multivariate_multistep)


plt.plot(y_test_multivariate_multistep[:, 0], color="black", label=f"Actual {COMPANY} Price")  
plt.plot(predicted_closing_prices[:, 0], color="green", label=f"Predicted {COMPANY} Price") 
plt.title(f"{COMPANY} Multivariate Multistep Share Price")
plt.xlabel("Time")
plt.ylabel(f"{COMPANY} Share Price")
plt.legend()
plt.show()



for i in range(steps_out):
    print(f"Predicted price for Day {i+1}: {predicted_closing_prices[0][i]}")
