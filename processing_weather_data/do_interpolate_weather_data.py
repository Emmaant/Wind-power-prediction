from helpers import *
import argparse
import os
import pandas as pd
from box import Box
import yaml


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Interpolate weather data to desired time interval")
    parser.add_argument('-config_path', type=str, help="Specify file path to the config file", default = 'processing_weather_data/config.yml')
    
    args = parser.parse_args()
    config = Box.from_yaml(filename=args.config_path, Loader=yaml.FullLoader)

    if config.file_path is None:
        print("Error: No file path specified. Please provide a file path in the config file")
        exit(1)

    # Check if the file exists
    if not os.path.exists(config.file_path):
        print(f"Error: The file '{config.file_path}' does not exist.")
        exit(1)

    # Check if the file has the correct extension
    if not config.file_path.endswith('.csv'):
        print(f"Error: The file '{config.file_path}' does not have a .csv extension. Please provide a valid csv file.")
        exit(1)

    try:
        df = pd.read_csv(config.file_path)
        df.rename(columns={df.columns[0]: 'time'}, inplace=True)
        df.set_index('time', inplace=True)
        print(f"Successfully opened the file: {config.file_path}")

    except Exception as e:
        print(f"Error: Could not open the file '{config.file_path}'. The file may be corrupted or not in the correct csv format.")
        print(f"Details: {e}")
        exit(1)


    df = interpolate_weather_data(df,config)
    full_file_name =  config.file_path_save +'.parquet'
    
    df.reset_index(inplace=True)
    df.rename(columns={df.columns[0]: 'time'}, inplace=True)
    
    df.to_parquet(full_file_name)
    print('File saved successfully with name: ', full_file_name)
    
