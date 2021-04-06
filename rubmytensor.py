#! /usr/bin/env python

import sys
import math
import numpy as np
import pandas as pd
import pandas_datareader as web
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')
pd.set_option('max_rows', 50)
pd.set_option('display.max_rows', 50)
pd.set_option('min_rows', 30)
pd.set_option('max_columns', None)
pd.set_option('display.max_columns', None)

if len(sys.argv) < 2:
  print("Usage:")
  print()
  print(f"{sys.argv[0]} <stock ticker> <y or n to display plot>")
  print()
  exit(1)

stock_ticker = sys.argv[1]
plot_graph = False
if len(sys.argv) >= 3 and sys.argv[2] == 'y':
  plot_graph = True

# For historical data end points - note that ideally, we would use all of this
# data to train with, and then estimate the next day's Close, but here we're
# doing some back testing prior to simply estimating.
start_date = '2012-01-01'
end_date = '2021-04-01'

# Play with training days because this is where potentially shorter intervals
# may yield better predictions.
days_training = 30

# This menas we'll only use a portion of the data to train with, and the
# remaining to do backtesting.
percent_of_data_to_train = .80

# Retrieve a dataframe of stock history, although we really only need the Close
df = web.DataReader(stock_ticker, data_source='yahoo', start=start_date, end=end_date)

# Create a new dataframe with only the 'Close' column
data = df.filter(['Close'])
dataset = data.values

# Get the number of rows to train the model on
training_data_len = math.ceil(len(dataset) * percent_of_data_to_train)

# Scale the data so that each data point is a value between 0 and 1.  Honestly,
# this isn't necessary and maybe even shouldn't be done here, but it's recommended
# that any time you train a new model, you should scale the data.  Some moron 
# wanted to make this more complicated.
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

# Create the training data set
train_data = scaled_data[0:training_data_len, :]
x_train = []
y_train = []

for i in range(days_training, len(train_data)):
  x_train.append(train_data[i - days_training:i, 0])
  y_train.append(train_data[i, 0])

# Convert the x_train and y_train to numpy array
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the data (LSTM model is expecting 3 dimensional data)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Build the LSTM model layers, and I see people across the web adding MANY of these
# layers, while LSTM is an RNN, and even though LSTM solves some of the gradient issues
# inherent in RNNs, I still think the fewer layers the better, unless the data in them
# truly significant.
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(20))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model (epochs may need to be adjusted to > 1)
model.fit(x_train, y_train, batch_size=1, epochs=1)

# Now back test the model
# Create a new array containing scaled values from the tested index to 100%
test_data = scaled_data[training_data_len - days_training: , :]

# Create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len: , :]
for i in range(days_training, len(test_data)):
  x_test.append(test_data[i - days_training:i, 0])

# Convert the data to a numpy array
x_test = np.array(x_test)

# Reshape the data (need it to be 3 dimensional)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Get the models predicted price values
predictions = model.predict(x_test)

# Unscale the values
predictions = scaler.inverse_transform(predictions)

# Get the root mean squared error (RMSE)
# ...evaluating performance of model based on standard deviation
rmse = np.sqrt(np.mean(((predictions - y_test)**2)))

# A value of 0.0 means that the values matched up perfectly.  You won't get that
# but anything between 1.0 and 5.0 is still good, but lower is better, up to 10
# is ok.  We can re-run the analysis repeatedly until we get a good score, if
# necessary.
if rmse > 10:
  print(f" [*] Considering predictions failed, rmse is: {rmse}")
  exit(1)

# Record final predictions for output
valid = data[training_data_len:]
valid['Predictions'] = predictions

# Output as text
print(f" [*] RMSE is: {rmse}")
print(f" [!] Stock predictions for {stock_ticker}")
print(valid)

# Visualize the data if requested
if plot_graph:
  train = data[:training_data_len]
  plt.figure(figsize=(16,8))
  plt.title('Model')
  plt.xlabel('Date', fontsize=18)
  plt.ylabel('Closing Price USD ($)', fontsize=14)
  plt.plot(train['Close'])
  plt.plot(valid[['Close', 'Predictions']])
  plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
  plt.show()

exit(0)
