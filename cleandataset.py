import pandas as pd
import numpy as np

vehicles = pd.read_csv("datasets/vehicles1.csv")

vehicle_id = pd.read_csv('datasets/vehicles1.csv', usecols=['vehicle_id'])
vehicle_id = vehicle_id.squeeze().tolist()
timestamp = pd.read_csv('datasets/vehicles1.csv', usecols=['timestamp'])
timestamp = timestamp.squeeze().tolist()
latitude = pd.read_csv('datasets/vehicles1.csv', usecols=['latitude'])
latitude = latitude.squeeze().tolist()
longitude = pd.read_csv('datasets/vehicles1.csv', usecols=['longitude'])
longitude = longitude.squeeze().tolist()

grouped_vehicles = vehicles.groupby('vehicle_id').apply(
    lambda x: list(zip(x['latitude'], x['longitude']))
).reset_index(name="coords")

grouped_vehicles.to_csv('datasets/grouped_vehicles.csv', sep='\t', encoding='utf-8', index=False, header=True)
