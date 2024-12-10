import pandas as pd
import numpy as np

vehicles = pd.read_csv("datasets/vehicles1.csv")

filtered_data = vehicles[['vehicle_id', 'latitude', 'longitude']]

filtered_data.to_csv('datasets/filtered_vehicles.csv', index=False)