#%%
# imports

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Input, Dropout, Bidirectional, Masking, RepeatVector, TimeDistributed
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.preprocessing.sequence import pad_sequences


scaler = MinMaxScaler(feature_range=(0, 1))

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

from itertools import islice

def pad_or_trim(features, seq_length, pad_value=0):
    """Ensure each list has exactly seq_length elements."""
    return list(islice(features, seq_length)) + [pad_value] * max(0, seq_length - len(features))

# %%
# preprocessing
def preprocessing(file, ratio_train=0.8, seq_length=None):
    # load dataset
    dataframe = pd.read_csv(file)
    grouped = dataframe.groupby('trajet_id').agg(
        num_coordinates=('latitude', 'size'),  # Number of coordinates for each vehicle
        features=('latitude', lambda lat: list(zip(lat, dataframe.loc[lat.index, 'longitude'])))  # Pair lat/lon as features
    ).reset_index()

    # Converting to numpy array
    vehicle_data = grouped.to_records(index=False)

    # Extract the list of features (coordinates) for all vehicles
    features_list = [vehicle['features'] for vehicle in vehicle_data]
    # Determine the maximum number of timestamps across vehicles
    if seq_length is None:
        seq_length = max(len(features) for features in features_list)
    
    # Ensure each list has exactly seq_length elements
    padded_features = [pad_or_trim(features, seq_length) for features in features_list]

    # Convert to a 3D NumPy array
    dataset = np.array(padded_features)

    # normalize the data
    # Assuming lstm_input has shape (num_vehicles, timestamps, 2)
    num_vehicles, timestamps, features = dataset.shape

    # Reshape to 2D: Combine vehicles and timestamps
    reshaped_data = dataset.reshape(-1, features)

    # Apply MinMaxScaler

    scaled_data = scaler.fit_transform(reshaped_data)

    # Reshape back to 3D
    dataset = scaled_data.reshape(num_vehicles, timestamps, features)

    # split dataset into train and test set
    trainSize = int(len(dataset)*ratio_train)   # 80% of dataset for training
    testSize = len(dataset) - trainSize # the rest (20%) for testing
    trainset = dataset[0:trainSize]
    testset = dataset[trainSize:]

    trainX, trainY = split_dataset(trainset)
    testX, testY = split_dataset(testset)

    return trainX, trainY, testX, testY

# %%
# implementation        
def simple_lstm_model(trainX,trainY,testX,testY, lstm_layers = 1, lstm_cells=64, epochs= 50, batch_size=64, validation_split=0.1):

    inputs_x = Input(shape=(trainX.shape[1], trainX.shape[2]))
    # Add the Masking layer
    masked_x = Masking(mask_value=0.0)(inputs_x)
    # Add the LSTM layer to ignore (0,0) padded values
    outputs_y = LSTM(lstm_cells, return_sequences=True)(masked_x)
    for _ in range(lstm_layers-1):
        outputs_y = LSTM(lstm_cells, return_sequences=True)(outputs_y)
    # Add the Dense output layer
    outputs_y = Dense(trainX.shape[2])(outputs_y)
    # create the model
    model = Model(inputs=inputs_x, outputs=outputs_y)

    model.compile(loss='mse', optimizer='adam')
    model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=1)

    return model

def encoder_decoder_model(trainX,trainY,testX,testY, encoder_lstm_cells=64, decoder_lstm_cells=64, epochs= 50, batch_size=64, validation_split=0.1):

    # Définir l'encodeur
    encoder_inputs = Input(shape=(trainX.shape[1], trainX.shape[2]))
    encoder_lstm = LSTM(encoder_lstm_cells, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)

    # Définir le décodeur
    decoder_inputs = Input(shape=(trainX.shape[1], trainX.shape[2]))
    decoder_lstm = LSTM(decoder_lstm_cells, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=[state_h, state_c])
    decoder_dense = Dense(trainX.shape[2])  # Prédire latitude et longitude
    decoder_outputs = decoder_dense(decoder_outputs)

    # Créer le modèle
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    # Compilation du modèle 
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    decoder_input_train = np.zeros_like(trainY)
    decoder_input_test = np.zeros_like(testY)

    model.fit(
        [trainX, decoder_input_train], trainY,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split
    )

    return model, decoder_input_train, decoder_input_test


def encoder_decoder_bidirectional_model(trainX,trainY,testX,testY, encoder_lstm_cells=64, decoder_lstm_cells=64, epochs= 50, batch_size=64, validation_split=0.1):

    # 6. Construction du modèle encodeur-décodeur bidirectionnel
    input_shape = (trainX.shape[1], trainX.shape[2])  # Forme d'entrée : (séquence, caractéristiques)
    output_shape = (trainY.shape[1], trainY.shape[2])  # Forme de sortie : (prédiction, caractéristiques)
    
    # Encodeur
    model = Sequential()
    model.add(Bidirectional(LSTM(encoder_lstm_cells, return_sequences=False, input_shape=input_shape)))  # Couche LSTM bidirectionnelle
    model.add(RepeatVector(output_shape[0]))  # Répéter le vecteur pour le décodeur

    # Décodeur
    model.add(Bidirectional(LSTM(decoder_lstm_cells, return_sequences=True)))  # Couche LSTM bidirectionnelle
    model.add(TimeDistributed(Dense(output_shape[1])))  # Couche dense pour chaque pas de temps

    # Compilation du modèle
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    # 7. Entraînement du modèle
    model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

    return model
# %%
# predictions

def prediction(model, trainX, trainY, testX, testY, decoder_input_train=None, decoder_input_test=None):
    # predict test values
    if decoder_input_train is None and decoder_input_test is None:
        trainPredict = model.predict(trainX)
        testPredict = model.predict(testX)
    else:    
        trainPredict = model.predict([trainX, decoder_input_train])
        testPredict = model.predict([testX, decoder_input_test])
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
    min_len = min(len(testY_flat), len(testPredict_flat))
    testY_flat = testY_flat[:min_len]
    testPredict_flat = testPredict_flat[:min_len]


    # Calculate RMSE
    trainScore = np.sqrt(mean_squared_error(trainY_flat, trainPredict_flat))
    print('Train Score: %.2f RMSE' % (trainScore))

    testScore = np.sqrt(mean_squared_error(testY_flat, testPredict_flat))
    print('Test Score: %.2f RMSE' % (testScore))

    return testPredict

# %%
# plotting

def plotting(testY, testPredict):
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

    # Plot True vs Predicted Latitude (first 100 values, with shifted testY)
    plt.figure(figsize=(10, 6))
    plt.plot(testY_flat[1:100, 0], label='True Latitude', alpha=0.8)
    plt.plot(testPredict_flat[:100, 0], label='Predicted Latitude', alpha=0.8)
    plt.legend()
    plt.title('True vs Predicted Latitude (First 100 Values)')
    plt.savefig("latfig") # save figure into image
    plt.show()
 

    # Plot True vs Predicted Longitude (first 100 values, with shifted testY)
    plt.figure(figsize=(10, 6))
    plt.plot(testY_flat[1:100, 1], label='True Longitude', alpha=0.8)
    plt.plot(testPredict_flat[:100, 1], label='Predicted Longitude', alpha=0.8)
    plt.legend()
    plt.title(' True vs Predicted Longitude (First 100 Values)')
    plt.savefig("longfig")  # save figure into image
    plt.show()

# %%
# savingCSV
def saveCSV(testY, testPredict):
    for i in range(testY.shape[0]):
        df = pd.DataFrame(testY[i], columns=['latitude', 'longitude'])
        df.to_csv(f"datasets/predictions/true_{i}.csv", index=False)
        df = pd.DataFrame(testPredict[i], columns=['latitude', 'longitude'])
        df.to_csv(f"datasets/predictions/predicted_{i}.csv", index=False)
    df.to_csv("results.csv", index=False)

# %%
# main

trainX, trainY, _, _ = preprocessing('datasets/outputs/cleaned_gpx.csv', seq_length=100)
_, _, testX, testY = preprocessing('datasets/test/cleaned_gpx_test.csv', ratio_train=0)
model = encoder_decoder_bidirectional_model(trainX, trainY, testX, testY)
testPredict = prediction(model, trainX, trainY, testX, testY)
plotting(testY,testPredict)

# %%
