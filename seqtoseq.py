#%%
# imports

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Input, Dropout, Masking   
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.preprocessing.sequence import pad_sequences

# %%
# load dataset

dataframe = pd.read_csv('datasets/filtered_vehicles.csv')
grouped = dataframe.groupby('vehicle_id').agg(
    num_coordinates=('latitude', 'size'),  # Number of coordinates for each vehicle
    features=('latitude', lambda lat: list(zip(lat, dataframe.loc[lat.index, 'longitude'])))  # Pair lat/lon as features
).reset_index()

# Converting to numpy array
vehicle_data = grouped.to_records(index=False)

# Extract the list of features (coordinates) for all vehicles
features_list = [vehicle['features'] for vehicle in vehicle_data]
# Determine the maximum number of timestamps across vehicles
max_timestamps = max(len(features) for features in features_list)

# Pad sequences to ensure all vehicles have the same number of timestamps
# Padding will add (0, 0) tuples where sequences are shorter than the maximum length
padded_features = pad_sequences(
    features_list,
    maxlen = max_timestamps,
    dtype='float32', 
    padding='post', 
    value=(0.0, 0.0)  # Default padding value
)

# Convert to a 3D NumPy array
dataset = np.array(padded_features)
dataset

# %%
# preprocessing

# normalize the data
# Assuming lstm_input has shape (num_vehicles, timestamps, 2)
num_vehicles, timestamps, features = dataset.shape

# Reshape to 2D: Combine vehicles and timestamps
reshaped_data = dataset.reshape(-1, features)

# Apply MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(reshaped_data)

# Reshape back to 3D
dataset = scaled_data.reshape(num_vehicles, timestamps, features)

# split dataset into train and test set
trainSize = int(len(dataset)*0.8)   # 80% of dataset for training
testSize = len(dataset) - trainSize # the rest (20%) for testing
trainset = dataset[0:trainSize]
testset = dataset[trainSize:]

def split_dataset(dataset):
    # Split into trainX and trainY
    trainX = dataset[:, :-1, :]  # All timestamps except the last
    trainY = dataset[:, 1:, :]   # All timestamps except the first (next point)
    return trainX, trainY

# split the dataset into X and Y arrays where X are t and Y t+1
def split_dataset_window(dataset, timesteps=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - timesteps):
        dataX.append(dataset[i:i+timesteps])  # Use 'timesteps' points as input
        dataY.append(dataset[i+timesteps])   # Predict the next point
    return np.array(dataX), np.array(dataY)

trainX, trainY = split_dataset(trainset)
testX, testY = split_dataset(testset)

trainX.shape

# %%
# implementation        

# create lstm network 
inputs_x = Input(shape=(trainX.shape[1], trainX.shape[2]))
# Add the Masking layer
masked_x = Masking(mask_value=0.0)(inputs_x)
# Add the LSTM layer to ignore (0,0) padded values
outputs_y = LSTM(64, return_sequences=True)(masked_x)
# Add the Dense layer
outputs_y = Dense(trainX.shape[2])(outputs_y)
# Define the model
model = Model(inputs=inputs_x, outputs=outputs_y)
# Print model summary
model.summary()

model.compile(loss='mse', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=16, validation_split=0.2, verbose=1)

# %%
# predictions

# predict test values
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# inverse normalisation
# Reshape the predictions and true values to 2D
trainPredict_reshaped = trainPredict.reshape(-1, trainPredict.shape[-1])
trainY_reshaped = trainY.reshape(-1, trainY.shape[-1])
testPredict_reshaped = testPredict.reshape(-1, testPredict.shape[-1])
testY_reshaped = testY.reshape(-1, testY.shape[-1])

# Apply inverse transformation
trainPredict_inverse = scaler.inverse_transform(trainPredict_reshaped)
trainY_inverse = scaler.inverse_transform(trainY_reshaped)
testPredict_inverse = scaler.inverse_transform(testPredict_reshaped)
testY_inverse = scaler.inverse_transform(testY_reshaped)

# Reshape back to 3D
trainPredict = trainPredict_inverse.reshape(trainPredict.shape)
trainY = trainY_inverse.reshape(trainY.shape)
testPredict = testPredict_inverse.reshape(testPredict.shape)
testY = testY_inverse.reshape(testY.shape)

# Calculate RMSE
# Reshape 3D arrays to 2D
trainY_flat = trainY.reshape(-1, trainY.shape[-1])
trainPredict_flat = trainPredict.reshape(-1, trainPredict.shape[-1])
testY_flat = testY.reshape(-1, testY.shape[-1])
testPredict_flat = testPredict.reshape(-1, testPredict.shape[-1])

# Calculate RMSE
trainScore = np.sqrt(mean_squared_error(trainY_flat, trainPredict_flat))
print('Train Score: %.2f RMSE' % (trainScore))

testScore = np.sqrt(mean_squared_error(testY_flat, testPredict_flat))
print('Test Score: %.2f RMSE' % (testScore))

# %%
# plotting
# Flatten the data to 2D
testY_flat = testY.reshape(-1, testY.shape[-1])
testPredict_flat = testPredict.reshape(-1, testPredict.shape[-1])

# Identify rows where all elements are [0.0, 0.0]
mask = np.all(testY_flat == 0.0, axis=1)
# Remove rows matching the condition
testY_flat = np.delete(testY_flat, np.where(mask), axis=0)

# Identify rows where all elements are [0.0, 0.0]
mask = np.all(testPredict_flat == 0.0, axis=1)
# Remove rows matching the condition
testPredict_flat = np.delete(testY_flat, np.where(mask), axis=0)


# Plot True vs Predicted Latitude
plt.figure(figsize=(10, 6))
plt.plot(testY_flat[:, 0], label='True Latitude', alpha=0.8)
plt.plot(testPredict_flat[:, 0], label='Predicted Latitude', alpha=0.8)
plt.legend()
plt.title('True vs Predicted Latitude')
plt.show()

# Plot True vs Predicted Longitude
plt.figure(figsize=(10, 6))
plt.plot(testY_flat[:, 1], label='True Longitude', alpha=0.8)
plt.plot(testPredict_flat[:, 1], label='Predicted Longitude', alpha=0.8)
plt.legend()
plt.title('True vs Predicted Longitude')
plt.show()

# %%

