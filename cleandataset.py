import pandas as pd
import numpy as np
import gpxpy

def clean_csv():
    vehicles = pd.read_csv("datasets/vehicles1.csv")

    filtered_data = vehicles[['vehicle_id', 'latitude', 'longitude']]

    filtered_data.to_csv('datasets/filtered_vehicles.csv', index=False)

def clean_gpx():
    gpx_path = 'datasets/gpxfilename.gpx'
    with open(gpx_path) as f:
        gpx = gpxpy.parse(f)

    data = []
    for segment in gpx.tracks[0].segments:
        for point in segment.points:
            data.append([point.latitude, point.longitude])
    df = pd.DataFrame(data, columns=['latitude', 'longitude'])
    df['vehicle_id'] = 1111
    df = df[['vehicle_id', 'latitude', 'longitude']]  # Reorder columns
    df.to_csv('datasets/filtered_gpx.csv', index=False)

clean_gpx()