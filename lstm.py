#%%
# imports

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

#%%
# load dataset

dataframe = pd.read_csv('datasets/passengers.csv',usecols=[1])
dataset = dataframe.values.astype('float32')

# %%
# preprocessing

# seed for reproductibility
tf.random.set_seed(7)
# normalize the data
scaler = MinMaxScaler(feature_range=(0,1))
dataset = scaler.fit_transform(dataset)
# split dataset into train and test set
trainSize = int(len(dataset)*0.8)   # 80% of dataset for training
testSize = len(dataset) - trainSize # the rest (20%) for testing
trainset = dataset[0:trainSize]
testset = dataset[trainSize:]
# split the dataset into X and Y arrays where X are t and Y t+1
def split_dataset(dataset, ):
    dataX, dataY = [], []
    for i in range(len(dataset)-1):
        dataX.append(dataset[i])
        dataY.append(dataset[i+1])
    return np.array(dataX), np.array(dataY) # convert them into array matrix

trainX, trainY = split_dataset(trainset)
testX, testY = split_dataset(testset)
#reshape inputs as [samples, timesteps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# %%
# implementation

# create lstm network 
model = Sequential()
model.add(tf.keras.Input((1,1)))    # input layer
model.add(LSTM(4))  # hidden layer
model.add(Dense(1)) # output layer 
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=1)

# %%
# predictions

# predict
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


# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[1:len(trainPredict)+1, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+1:len(dataset)-1, :] = testPredict

plt.plot(scaler.inverse_transform(dataset), c='green')
plt.plot(trainPredictPlot, c='blue')
plt.plot(testPredictPlot, c='red')
plt.legend(['dataset', 'training', 'prediction'])
plt.show()
# %%
