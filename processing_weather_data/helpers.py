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

def transform_periodic_data(df, col_names):
    df = df.copy()
    col_names = [col_names] if isinstance(col_names, str) else col_names
    for prefix in col_names:
        columns_list = [col for col in df.columns if col.startswith(prefix)]
        for col in columns_list:
            sin, cos = transform_sin_cos(df[col])
            df[f"sin_{col}"] = sin
            df[f"cos_{col}"] = cos
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

def interpolate_weather_data(df,config):
   df = df.copy()
   df.index = pd.to_datetime(df.index)
   df = transform_periodic_data(df, config.columns_transform_periodic)
   
   new_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq=config.interpolate_to_interval)
   df = df.reindex(new_index)
   
   df = linear_interpolation(df,config.columns_linear_interpolation)
   df = forward_and_backward_fill(df, config.columns_backward_forward_fill)
   
   for prefix in config.columns_transform_periodic:
        cols = [col for col in df.columns if col.startswith(f"sin_{prefix}")]
        for sin_col in cols:
            suffix = sin_col[len("sin_"):]
            cos_col = f"cos_{suffix}"
            orig_col = suffix
            if cos_col in df.columns:
                df[orig_col] = inverse_transform_sin_cos(df[sin_col], df[cos_col])
   return df

