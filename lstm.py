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
trainsize = int(len(dataset)*0.8)   # 80% of dataset for training
testsize = len(dataset) - trainsize # the rest (20%) for testing
trainset = dataset[0:trainsize]
testset = dataset[trainsize:]
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
