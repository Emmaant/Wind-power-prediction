"""
Script to create local features in_degree and out_degree for training ml models
Script is used to add in degree and out degree for each turbine with wake settings specified in config_data

Columns added are out_degree_(angle, base_width, length)_XXX, in_degree_(angle, base_width, length)_XXX
where _XXX denotes the specified turbine index
"""

from box import Box
import os
import yaml
import argparse
import pandas as pd
from itertools import product 

from topography_tools import *

def initialize_data(df,coordinates, config,wd_column):
    """
    Function to add columns indegree and out degree for graphs with different settings
    """
    wake_length = config.wake_settings.length
    wake_base_width = config.wake_settings.base_width
    wake_angle = config.wake_settings.angle
    parameter_combinations = list(product(wake_angle, wake_base_width, wake_length))

    result_dfs = []
    for parameters in parameter_combinations:
        in_degree_columns = [f"in_degree_{str(parameters)}_{i:03d}" for i in range(16)]
        out_degree_columns = [f"out_degree_{str(parameters)}_{i:03d}" for i in range(16)]
        all_columns = in_degree_columns + out_degree_columns

        new_data = df.apply(
            lambda row: get_in_out_degree(row, coordinates, parameters[2], parameters[1], parameters[0], wd_column),
            axis=1,
            result_type="expand"
        )

        new_data.columns = all_columns

        result_dfs.append(new_data)

    return pd.concat([df] + result_dfs, axis=1)

def convert_adjacency_matrix_to_edge_list(A):
    """
    Function which converts adjacency matrix to edge list
    """
    # Get non-zero element indicies
    edge_list = np.nonzero(A.T)

    return np.array(edge_list)

def get_edge_list(wd, coordinates, length, base_width, angle):
    """
    Function which gets wake fields for specified settings of wake and coverts adjacency matrix
    to edge list by calling function convert_adjacency_matrix_to_edge_list.

    Returns: 
        Edge list for specified wake settings
    """
    turbine_wakefields = get_wakefields(coordinates,wd, length, base_width, angle)
    A = get_wake_matrix(coordinates, turbine_wakefields)
    edge_index = convert_adjacency_matrix_to_edge_list(A)

    return edge_index

def get_in_out_degree(row,coordinates, length, base_width, angle,wd_column):
    """
    Function which calculates in degree and out degree from edge list
    """

    wd = row[wd_column]
    edge_list = get_edge_list(wd, coordinates, length, base_width, angle)

    in_degree = np.bincount(edge_list[0], minlength=16)
    out_degree = np.bincount(edge_list[1], minlength=16)
    return np.concatenate((in_degree, out_degree))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_config', '-c', help="Path to the YAML config file", type=str, required=True)
    parser.add_argument('--output_dir', '-o', help="Directory to save processed data", type=str, default='data/train_data/ml_data/')

    args = parser.parse_args()

    print(args.yaml_config,)

    config = Box.from_yaml(filename=args.yaml_config, Loader=yaml.FullLoader)

    # Get list of turbine coordinates
    windFarmCoord = pd.read_csv(config.wind_farm_path)
    x_pos, y_pos, z_pos = windFarmCoord[['X (m)', 'Y (m)', 'Z (m)']].to_numpy().T
    coordinates = np.array([x_pos, y_pos]).T

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load and create columns by calling initialize_data function
    if config.hourly_data:
        df = pd.read_parquet(config.data_paths_res_hour.scada_data)
        df = initialize_data(df, coordinates, config, 'wd')
        output_file = os.path.join(args.output_dir, 'df_ifs_scada_hourly.parquet')
        df.to_parquet(output_file)
        print('Hourly data saved successfully to', output_file)
    else:
        df = pd.read_parquet(config.data_paths.scada_data)
        df = initialize_data(df, coordinates, config, 'wd')
        output_file = os.path.join(args.output_dir, 'df_ifs_scada.parquet')
        df.to_parquet(output_file)
        print('Data saved successfully to', output_file)
