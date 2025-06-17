from helpers import *
import argparse
import os
import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Interpolate weather data to desired time interval")
    parser.add_argument('-file_path', type=str, help="Specify file path to .parquet file", default = 'data/IFS/global_weather_data_turbines.parquet')
    parser.add_argument('-file_name', type=str, help="Specify file name to save file as", default='IFS_data_interpolated')

    args = parser.parse_args()

    if args.file_path is None:
        print("Error: No file path specified. Please provide a file path using -file_path")
        exit(1)

    # Check if the file exists
    if not os.path.exists(args.file_path):
        print(f"Error: The file '{args.file_path}' does not exist.")
        exit(1)

    # Check if the file has the correct extension
    if not args.file_path.endswith('.parquet'):
        print(f"Error: The file '{args.file_path}' does not have a .parquet extension. Please provide a valid parquet file.")
        exit(1)

    try:
        df = pd.read_parquet(args.file_path)
        df.rename(columns={df.columns[0]: 'time'}, inplace=True)
        df.set_index('time', inplace=True)
        print(f"Successfully opened the file: {args.file_path}")

    except Exception as e:
        print(f"Error: Could not open the file '{args.file_path}'. The file may be corrupted or not in the correct parquet format.")
        print(f"Details: {e}")
        exit(1)


    df = interpolate_weather_data(df)
    full_file_name = 'data/' + args.file_name +'.parquet'
    
    df.reset_index(inplace=True)
    df.rename(columns={df.columns[0]: 'time'}, inplace=True)
    
    df.to_parquet(full_file_name)
    print('File saved successfully with name: ', full_file_name)
    
