import numpy as np
import pandas as pd


def transform_sin_cos(degrees):
    radians = np.deg2rad(degrees)
    return np.sin(radians), np.cos(radians)

def inverse_transform_sin_cos(sin_val, cos_val):
    radians = np.arctan2(sin_val, cos_val)
    degrees = np.rad2deg(radians)
    degrees = (degrees + 360) % 360  # Ensure degrees are in [0, 360)
    return degrees

def transform_periodic_data(df,col_name):
    df = df.copy()
    columns_list = [col for col in df.columns if col.startswith(col_name)]
    for col in columns_list:
        sin, cos = transform_sin_cos(df[col])
        sin_col = 'sin_' + col 
        cos_col = 'cos_' + col 
        df[sin_col] = sin
        df[cos_col] = cos
    return df

def linear_interpolation(df, cols_list):
    df = df.copy()
    columns_to_interpolate = [col for col_name in cols_list for col in df.columns if col.startswith(col_name)]
    df[columns_to_interpolate] = df[columns_to_interpolate].interpolate(method='linear')
    return df


def forward_and_backward_fill(df, columns):
    df = df.copy()
    columns_to_fill = [col for col_name in columns for col in df.columns if col.startswith(col_name)]
    for col in columns_to_fill:
        if col in df.columns:
            df[col] = df[col].ffill( limit=2)  
            df[col] = df[col].bfill( limit=3)  
    return df

def interpolate_weather_data(df,interval = '10min'):
   df = df.copy()
   df.index = pd.to_datetime(df.index)
   df = transform_periodic_data(df,'wind_direction_100m')
   
   new_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq=interval)
   df = df.reindex(new_index)
   
   df = linear_interpolation(df,["cos_wind_direction_100m","sin_wind_direction_100m","wind_speed_100m","temperature_2m","relative_humidity_2m","surface_pressure"])
   df = forward_and_backward_fill(df, ["precipitation", "snowfall", "rain"])

   for i in range(16):  
        sin_col = f'sin_wind_direction_100m_{i:03d}'  
        cos_col = f'cos_wind_direction_100m_{i:03d}'
        if sin_col in df.columns and cos_col in df.columns:
            wind_col = f'wind_direction_100m_{i:03d}'
            df[wind_col] = inverse_transform_sin_cos(df[sin_col], df[cos_col])
   return df

