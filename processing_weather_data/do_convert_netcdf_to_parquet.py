from helpers import *
import argparse
import os
import netCDF4 as nc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert weather data with format NetCDF to csv file")
    parser.add_argument('-file_path', type=str, help="Specify file path to .nc file")
    parser.add_argument('-file_name', type=str, help="Specify file name to save file as", default='ERA5_data')
    parser.add_argument('-data_type', type=str, choices = ['wind','temperature'], default='wind')

    args = parser.parse_args()

    if args.file_path is None:
        print("Error: No file path specified. Please provide a file path using -file_path")
        exit(1)

    # Check if the file exists
    if not os.path.exists(args.file_path):
        print(f"Error: The file '{args.file_path}' does not exist.")
        exit(1)

    # Check if the file has the correct extension
    if not args.file_path.endswith('.nc'):
        print(f"Error: The file '{args.file_path}' does not have a .nc extension. Please provide a valid NetCDF file.")
        exit(1)

    try:
        dataset = nc.Dataset(args.file_path)
        print(f"Successfully opened the file: {args.file_path}")

    except Exception as e:
        print(f"Error: Could not open the file '{args.file_path}'. The file may be corrupted or not in the correct NetCDF format.")
        print(f"Details: {e}")
        exit(1)

    if args.data_type == 'wind':
        dictionary = netcdf_to_df(dataset)
        convert_uv(dictionary,args.file_name)
    
    if args.data_type == 'temperature':
        netcdf_to_df_temp(dataset,args.file_name)
    




