import numpy as np
import pandas as pd
from datetime import datetime
import cftime
import itertools

def convert_to_datetime(dataset):
    """
    Converts time column with 
    units: seconds since 1970-01-01
    calendar: proleptic_gregorian 
    to date time format
    """

    valid_time = dataset.variables['valid_time'][:]

    # Get the time units and calendar
    time_units = dataset.variables['valid_time'].units
    calendar = dataset.variables['valid_time'].calendar

    # Convert to cftime.datetime objects
    valid_time_cftime = cftime.num2date(valid_time, units=time_units, calendar=calendar)

    # Convert cftime.datetime to standard datetime.datetime
    valid_time_dt = np.array([datetime(t.year, t.month, t.day, t.hour, t.minute) for t in valid_time_cftime])

    return valid_time_dt

def netcdf_to_df_temp(dataset,file_name):
    """
    Input: NetCDF file
    Returns: dictionary with time, latitude and longitude and u,v wind components
    """

    #Convering valid time to datetime format
    time = convert_to_datetime(dataset)

    #Initialize df with time
    dictionary = {'time': time,
                  'coordinates':{}}
    
    print(dataset.variables)

    latitude = dataset.variables['latitude'][:]  #Shape (latitude)
    longitude = dataset.variables['longitude'][:] #Shape (longitude)

    temperature = dataset.variables['t2m'][:] #Shape (valid time, latitude, longitude)

    #Extract u,v values for latitude and longitude
    for lat, long in itertools.product(latitude, longitude):

        lat_index = np.where(latitude == lat)[0][0]  # Find the index of lat
        lon_index = np.where(longitude == long)[0][0]  # Find the index of lon
        
        # Extract u and v components for the corresponding lat/lon indices
        temp = temperature[:, lat_index, lon_index].flatten()

        lat_long_str = f'({lat:.02f},{long:.02f})'

        dictionary['coordinates'][lat_long_str] = {'temperature':temp}

    df = pd.DataFrame({'time': dictionary['time']})

    for key, value in (dictionary['coordinates'].items()):
        df[f'temp_{key}'] = value['temperature']

        full_file_name = '../../data/' + file_name +'.parquet'

    df.to_parquet(full_file_name)
    print('File saved successfully with name: ', full_file_name)


def netcdf_to_df(dataset):
    """
    Input: NetCDF file
    Returns: dictionary with time, latitude and longitude and u,v wind components
    """

    #Convering valid time to datetime format
    time = convert_to_datetime(dataset)

    #Initialize df with time
    dictionary = {'time': time,
                  'coordinates':{}}
    
    latitude = dataset.variables['latitude'][:]  #Shape (latitude)
    longitude = dataset.variables['longitude'][:] #Shape (longitude)

    u100 = dataset.variables['u100'][:] #Shape (valid time, latitude, longitude)
    v100 = dataset.variables['v100'][:] #Shape (valid time, latitude, longitude)

    #Extract u,v values for latitude and longitude
    for lat, long in itertools.product(latitude, longitude):

        lat_index = np.where(latitude == lat)[0][0]  # Find the index of lat
        lon_index = np.where(longitude == long)[0][0]  # Find the index of lon
        
        # Extract u and v components for the corresponding lat/lon indices
        u = u100[:, lat_index, lon_index].flatten()
        v = v100[:, lat_index, lon_index].flatten()

        lat_long_str = f'({lat:.02f},{long:.02f})'

        dictionary['coordinates'][lat_long_str] = {'u':u, 'v':v}
        
    return dictionary


def calculate_ws_wd(u, v):
    """
    Input: u,v wind components
    Returns: Wind speed and wind direction calculated from u,v components
    """
    #Calculate wind speed from u, v components
    wind_speed = np.sqrt(u**2 + v**2)
    
    #Calculate wind direction from u, v components
    wind_direction = (180 + 180/np.pi * np.arctan2(u, v)) % 360
    
    return wind_speed, wind_direction


def convert_uv(dictionary,file_name):
    """
    Function which takes df with u,v components for latitude and longitude values
    calculates wind speed and wind direction and saves it as a parquet file.
    """

    df = pd.DataFrame({'time': dictionary['time']})

    for key, value in (dictionary['coordinates'].items()):
  
        ws, wd = calculate_ws_wd(value['u'], value['v'])

        df[f'ws_{key}'] = ws
        df[f'wd_{key}'] = wd

    full_file_name = '../../data/' + file_name +'.parquet'

    df.to_parquet(full_file_name)
    print('File saved successfully with name: ', full_file_name)

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

