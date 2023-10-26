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

#------------------------------------------------------------------------------
# Load Data
## TO DO:
# 1) Check if data has been saved before. 
# If so, load the saved data
# If not, save the data into a directory
#------------------------------------------------------------------------------
DATA_SOURCE = "yahoo"
COMPANY = "TSLA"
DATA_FILE = "TSLA_data.csv"

# start = '2012-01-01', end='2017-01-01'
TRAIN_START = '2015-01-01'
TRAIN_END = '2020-01-01'

#Train start and end date validation -- Train start date must be earlier than train end date
def start_end_date (TRAIN_START_STR ='2025-01-01', TRAIN_END_STR = '2020-01-01', date_format = "%Y-%m-%d"):
    train_start = datetime.strptime(TRAIN_START_STR, date_format)
    train_end = datetime.strptime(TRAIN_END_STR, date_format)
    print(train_start,train_end)
    if train_start < train_end:
        print("Error: Train start date must be earlier than train end date!")
    return train_start, train_end

# print(start_end_date(TRAIN_START,TRAIN_END))

data =  yf.download(COMPANY, start=TRAIN_START, end=TRAIN_END, progress=False)
# yf.download(COMPANY, start = TRAIN_START, end=TRAIN_END)

#Deal with NaN issue with forward fill method
def handle_missing_values(df):
    if df.isnull().sum().sum() > 0:
        df.fillna(method='ffill', inplace=True)
    return df

data = handle_missing_values(data)

#https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split
def custom_train_test_split(data, split_ratio=0.8, method='random'):
    
    train_data, test_data = train_test_split(data, train_size=split_ratio, random_state=42)
    
    return train_data, test_data

train, test = custom_train_test_split(data, split_ratio=0.8, method='random')

print(custom_train_test_split(data, split_ratio=0.8, method="random"))

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

print(data_to_file(DATA_FILE))

PRICE_VALUE = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

scaler = MinMaxScaler(feature_range=(0, 1)) 

scaled_data = scaler.fit_transform(data[PRICE_VALUE].values) 

print(scaled_data)

# Number of days to look back to base the prediction
PREDICTION_DAYS = 60 # Original

# To store the training data
x_train = []
y_train = []

scaled_data = scaled_data[:,0] # Turn the 2D array back to a 1D array
# Prepare the data
for x in range(PREDICTION_DAYS, len(scaled_data)):
    x_train.append(scaled_data[x-PREDICTION_DAYS:x])
    y_train.append(scaled_data[x])

x_test = []
y_test = []

# scaled_data = scaled_data[:,0] # Turn the 2D array back to a 1D array
# Prepare the data
for x in range(len(scaled_data), len(data)):
    x_test.append(scaled_data[x-PREDICTION_DAYS:x])
    y_test.append(scaled_data[x])
    
    
def create_array(a, b):
    a, b = np.array(a), np.array(b)
    return a, b
x_train, y_train = create_array(x_train, y_train)
x_test, y_test = create_array(x_test, y_test)
# Convert them into an array
# x_train, y_train = np.array(x_train), np.array(y_train)
# Now, x_train is a 2D array(p,q) where p = len(scaled_data) - PREDICTION_DAYS
# and q = PREDICTION_DAYS; while y_train is a 1D array(p)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

def custom_model(sequence_length, n_features, layer_type=LSTM, n_layers=2, layer_units=50,
                        dropout=0.3, loss="mean_absolute_error", optimizer="rmsprop", bidirectional=False):
  
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
    model.add(Dense(1, activation="linear"))

    # Compile the model
    model.compile(loss=loss, metrics=["mean_absolute_error"], optimizer=optimizer)

    return model

sequence_length = 60 
n_features = 1  

# Create a custom LSTM model
lstm_model = custom_model(sequence_length, n_features, layer_type=SimpleRNN, n_layers=2, 
                                 layer_units=50, dropout=0.2, optimizer="adam")

lstm_model.fit(x_train, y_train, epochs=25, batch_size=32)

#------------------------------------------------------------------------------
# Test the model accuracy on existing data
#------------------------------------------------------------------------------
# Load the test data
TEST_START = '2020-01-02'
TEST_END = '2022-12-31'

test_data = yf.download(COMPANY, start=TRAIN_START, end=TRAIN_END, progress=False)

# The above bug is the reason for the following line of code
test_data = test_data[1:]

actual_prices = test_data[PRICE_VALUE].values

total_dataset = pd.concat((data[PRICE_VALUE], test_data[PRICE_VALUE]), axis=0)

# model_inputs = total_dataset[len(total_dataset) - len(test_data) - PREDICTION_DAYS:].values
model_inputs = test_data[-1]

# model_inputs = model_inputs.reshape(-1, 1)
model_inputs = np.reshape(model_inputs, (1, sequence_length, len(feature_columns)))

# TO DO: Explain the above line

model_inputs = scaler.transform(model_inputs)

#------------------------------------------------------------------------------
# Make predictions on test data
#------------------------------------------------------------------------------
x_test = []
for x in range(PREDICTION_DAYS, len(model_inputs)):
    x_test.append(model_inputs[x - PREDICTION_DAYS:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
# TO DO: Explain the above 5 lines

predicted_prices = lstm_model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)
# Clearly, as we transform our data into the normalized range (0,1),
# we now need to reverse this transformation 
#------------------------------------------------------------------------------
# Plot the test predictions
## To do:
# 1) Candle stick charts
# 2) Chart showing High & Lows of the day
# 3) Show chart of next few days (predicted)
#------------------------------------------------------------------------------

plt.plot(actual_prices, color="black", label=f"Actual {COMPANY} Price")
plt.plot(predicted_prices, color="green", label=f"Predicted {COMPANY} Price")
plt.title(f"{COMPANY} Share Price")
plt.xlabel("Time")
plt.ylabel(f"{COMPANY} Share Price")
plt.legend()
plt.show()

#------------------------------------------------------------------------------
# Predict next day
#------------------------------------------------------------------------------


real_data = [model_inputs[len(model_inputs) - PREDICTION_DAYS:, 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

prediction = lstm_model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(f"Prediction: {prediction}")
