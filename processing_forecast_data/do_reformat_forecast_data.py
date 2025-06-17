import glob
import argparse
import os
import pandas as pd
import pickle
import numpy as np

"""
This script merges forecast data into the SCADA dataset and saves the data according to train, validation and test index. 
One df for each.

The following forecast columns are added:
- 'ws_for'      : Forecasted wind speed
- 'wd_for'      : Forecasted wind direction
- 'wd_for_cos'  : Forecasted wind direction (cosine-transformed)
- 'wd_for_sin'  : Forecasted wind direction (sine-transformed)
- 'temp_for'    : Forecasted temperature
- 'rh_for'      : Forecasted relative humidity
- 'sap_for'     : Forecasted surface air pressure
- 'time_diff'   : Forecast lead time (set to 5 for all forecasts)

Note:
- 'time_diff' indicates the lead time between the forecast issue and the target time 'time'. 

If forecast data was extracted for longer than 6h lead times. This means duplicate times will be availible in dataframe.

"""


def get_index(path):
        """Read in data split index from pickle file."""
        index = []
        try:
            with open(path, "rb") as openfile:
                while True:
                    try:
                        index.append(pickle.load(openfile))
                    except EOFError:
                        break
        except FileNotFoundError:
            print(f"File not found: {path}")
        return index

def add_avg_weather_features(df, weather_columns):
    for col in weather_columns:
        matching_cols = df.filter(regex=col).columns
        if matching_cols.empty:
            print(f"Error: No columns found matching '{col}'")
            continue
        
        df[f'{col}_avg'] = df[matching_cols].mean(axis=1)

    return df

def add_cyclic_features(df, column_prefixes, n_turbines=None):
    if isinstance(column_prefixes, str):
        column_prefixes = [column_prefixes]

    for prefix in column_prefixes:
        if n_turbines is None:
            # Single column case
            if prefix in df.columns:
                radians = np.deg2rad(df[prefix])
                df[f'{prefix}_cos'] = np.cos(radians)
                df[f'{prefix}_sin'] = np.sin(radians)
        else:
            # Multi-turbine case
            for i in range(n_turbines):
                col_name = f'{prefix}{str(i).zfill(3)}'
                if col_name in df.columns:
                    radians = np.deg2rad(df[col_name])
                    df[f'{prefix.strip()}cos_{str(i).zfill(3)}'] = np.cos(radians)
                    df[f'{prefix.strip()}sin_{str(i).zfill(3)}'] = np.sin(radians)
    return df

def add_time_features(df):
    df = df.copy()  
    
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], utc=True)
        
        df['time_of_day'] = df['time'].dt.hour
        df['sin_time_of_day'] = np.sin(2 * np.pi * df['time_of_day'] / 24)
        df['cos_time_of_day'] = np.cos(2 * np.pi * df['time_of_day'] / 24)
        
        # Extract month (1-12)
        df['month_of_year'] = df['time'].dt.month
        
        # Add cyclic features for month (1-12)
        df['sin_month_of_year'] = np.sin(2 * np.pi * df['month_of_year'] / 12)
        df['cos_month_of_year'] = np.cos(2 * np.pi * df['month_of_year'] / 12)

    else:
        print('Time column not present')
    
    return df


def inverse_transform_sin_cos(sin_val, cos_val):
    radians = np.arctan2(sin_val, cos_val)
    degrees = np.rad2deg(radians)
    degrees = (degrees + 360) % 360  # Ensure degrees are in range [0, 360)
    return degrees


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run to get forcast data frame merged with test data")
    parser.add_argument('--forcast_dir', '-d', help="Path to the forecast data directory, expected format directory of .parquet files", type=str,required=True)
    parser.add_argument('--scada_data', '-f', help="Path to the scada data ", type=str , default= 'data/train_data/scada_ifs_filtered_hourly.parquet')
    parser.add_argument('--index_path', '-i', help="Path to the .pkl index file ", type=str, default='data/train_data/ml_data/index_hourly.pkl')

    args = parser.parse_args()

    #Reading forecast data and concatenate to DataFrame
    dir_forcast = args.forcast_dir
    parquet_files = glob.glob(os.path.join(dir_forcast, '*.parquet'))
    df = pd.concat((pd.read_parquet(file) for file in parquet_files), ignore_index=True)

    #Add turbine index to DataFrame
    df_farm = pd.read_csv('data/coordinates_kronoberg.csv')
    unique_pairs = df_farm[['Latitude','Longitude','Item']]
    unique_pairs = unique_pairs.rename(columns={"Latitude": "latitude", "Longitude": "longitude"})
    df = df.merge(unique_pairs, on=['latitude', 'longitude'], how='left')
    df['Item'] = df['Item'].apply(lambda x: str(int(x)).zfill(3) if pd.notna(x) else x)

    #Removing and renaming columns
    df = df.drop(columns=['latitude', 'longitude', 'u', 'v'])
    df = df.rename(columns={'wind_speed': 'ws_for', 'wind_direction': 'wd_for', 'surface_air_pressure':'sap_for', 
                            'temperature':'temp_for','relative_humidity':'rh_for'})
    
    #Changing units to match ifs data
    temp_cols = [col for col in df.columns if col.startswith('temp_for')]
    df[temp_cols] = df[temp_cols] - 273.15

    sap_cols = [col for col in df.columns if col.startswith('sap_for')]
    df[sap_cols] = df[sap_cols]/ 100

    # Melting df
    value_vars = ['ws_for', 'wd_for', 'temp_for', 'rh_for', 'sap_for']
    df_long = df.melt(
        id_vars=['forecast_time', 'Item', 'forecast_run'],
        value_vars=value_vars,
        var_name='variable',
        value_name='value'
    )

    # Create a new column that combines variable and Item
    df_long['variable_item'] = df_long['variable'] + '_' + df_long['Item']

    # Pivot so each row is a unique forecast_time
    df_wide = df_long.pivot_table(
        index=['forecast_time','forecast_run'],
        columns='variable_item',
        values='value'
    ).reset_index()

    df_wide.columns.name = None
    df_wide.columns = [str(col) for col in df_wide.columns]

    #Remove time zone
    df_wide['forecast_run'] = pd.to_datetime(df_wide['forecast_run']).dt.tz_localize(None)
    df_wide['forecast_time'] = pd.to_datetime(df_wide['forecast_time']).dt.tz_localize(None)
    
    #Calculate forcast time
    df_wide['time_diff'] = ((df_wide['forecast_time'] - df_wide['forecast_run']).dt.total_seconds() // 3600).astype(int)

    path_index = args.index_path
    index = get_index(path_index)

    # Read in filtered ifs scada data
    path_scada = args.scada_data
    df_scada_ifs = pd.read_parquet(path_scada)

    #Getting train data
    df_scada_ifs_train = df_scada_ifs.iloc[index[0][0]].copy()
    df_scada_ifs_train['time'] = df_scada_ifs_train['time'].dt.tz_localize(None)

    df_merged_train = df_scada_ifs_train.merge(
        df_wide,
        left_on='time',
        right_on='forecast_time',
        how='inner'
    )

    print(len(df_merged_train))

    #Average forcast wind speed, average temperature, och average direction
    df_merged_train = add_cyclic_features(df_merged_train, ['wd_for_'], 16)
    df_merged_train = add_cyclic_features(df_merged_train, ['wd_for_avg'])
    df_merged_train = add_time_features(df_merged_train)
    weather_columns = ['ws_for', 'wd_for_cos', 'wd_for_sin', 'temp_for', 'rh_for', 'sap_for']
    df_merged_train = add_avg_weather_features(df_merged_train, weather_columns)
    df_merged_train['wd_for_avg'] = inverse_transform_sin_cos(df_merged_train['wd_for_sin_avg'], df_merged_train['wd_for_cos_avg'])


    df_merged_train.to_parquet('data/scada_ifs_forecast_train.parquet')
    print('Successfully saved data/scada_ifs_forecast_train.parquet')

    #Getting validation data
    df_scada_ifs_val = df_scada_ifs.iloc[index[0][1]].copy()
    df_scada_ifs_val['time'] = df_scada_ifs_val['time'].dt.tz_localize(None)

    df_merged_val = df_scada_ifs_val.merge(
        df_wide,
        left_on='time',
        right_on='forecast_time',
        how='inner'
    )

    print(len(df_merged_val))

    #Average forcast wind speed, average temperature, och average direction
    df_merged_val = add_cyclic_features(df_merged_val, ['wd_for_'], 16)
    df_merged_val = add_cyclic_features(df_merged_val, ['wd_for_avg'])
    df_merged_val = add_time_features(df_merged_val)
    weather_columns = ['ws_for', 'wd_for_cos', 'wd_for_sin', 'temp_for', 'rh_for', 'sap_for']
    df_merged_val = add_avg_weather_features(df_merged_val, weather_columns)
    df_merged_val['wd_for_avg'] = inverse_transform_sin_cos(df_merged_val['wd_for_sin_avg'], df_merged_val['wd_for_cos_avg'])


    df_merged_val.to_parquet('data/scada_ifs_forecast_val.parquet')
    print('Successfully saved data/scada_ifs_forecast_val.parquet')

    #Getting test data
    df_scada_ifs_test = df_scada_ifs.iloc[index[0][2]].copy()
    df_scada_ifs_test['time'] = df_scada_ifs_test['time'].dt.tz_localize(None)

    df_merged_test = df_scada_ifs_test.merge(
        df_wide,
        left_on='time',
        right_on='forecast_time',
        how='inner'
    )

    print(len(df_merged_test))

    #Average forcast wind speed, average temperature, och average direction
    df_merged_test = add_cyclic_features(df_merged_test, ['wd_for_'], 16)
    df_merged_test = add_cyclic_features(df_merged_test, ['wd_for_avg'])
    df_merged_test = add_time_features(df_merged_test)
    weather_columns = ['ws_for', 'wd_for_cos', 'wd_for_sin', 'temp_for', 'rh_for', 'sap_for']
    df_merged_test = add_avg_weather_features(df_merged_test, weather_columns)
    df_merged_test['wd_for_avg'] = inverse_transform_sin_cos(df_merged_test['wd_for_sin_avg'], df_merged_test['wd_for_cos_avg'])


    df_merged_test.to_parquet('data/scada_ifs_forecast_test.parquet')
    print('Successfully saved data/scada_ifs_forecast_test.parquet')

