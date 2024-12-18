#%%
# imports

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout     
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from matrepr import mdisplay

# %%
# load dataset

dataframe = pd.read_csv('datasets/filtered_vehicles.csv')
subset = dataframe.query('vehicle_id == 2406')
subset = subset.drop(columns=['vehicle_id'])
dataset = subset.astype('float32').values
dataset.shape
# %%
# preprocessing

# normalize the data
scaler = MinMaxScaler(feature_range=(0,1))
dataset = scaler.fit_transform(dataset)
# split dataset into train and test set
trainSize = int(len(dataset)*0.8)   # 80% of dataset for training
testSize = len(dataset) - trainSize # the rest (20%) for testing
trainset = dataset[0:trainSize]
testset = dataset[trainSize:]
# split the dataset into X and Y arrays where X are t and Y t+1
def split_dataset(dataset, timesteps=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - timesteps):
        dataX.append(dataset[i:i+timesteps])  # Use 'timesteps' points as input
        dataY.append(dataset[i+timesteps])   # Predict the next point
    return np.array(dataX), np.array(dataY)

trainX, trainY = split_dataset(trainset, timesteps=1)
testX, testY = split_dataset(testset, timesteps=1)

mdisplay(trainY, floatfmt=".2f", max_rows=18, max_cols=3)

# Reshape inputs for LSTM [, timesteps, features]
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], trainX.shape[2]))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], testX.shape[2]))

# %%
# implementation

# create lstm network 
model = Sequential()
model.add(LSTM(50, input_shape=(trainX.shape[1], trainX.shape[2])))
model.add(Dense(2))
model.compile(loss='mse', optimizer='adam')

model.fit(trainX, trainY, epochs=100, batch_size=16, validation_split=0.2, verbose=1)

# %%
# predictions

# predict test values
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# inverse normalisation
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform(trainY)
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform(testY)
# calculate error
trainScore = np.sqrt(mean_squared_error(trainY, trainPredict))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = np.sqrt(mean_squared_error(testY, testPredict))
print('Test Score: %.2f RMSE' % (testScore))

# %%
# plotting
plt.figure(figsize=(10, 6))
plt.plot(testY[:, 0], label='True Latitude')
plt.plot(testPredict[:, 0], label='Predicted Latitude')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(testY[:, 1], label='True Longitude')
plt.plot(testPredict[:, 1], label='Predicted Longitude')
plt.legend()
plt.show()

# %%
