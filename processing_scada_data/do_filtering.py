import numpy as np
import pandas as pd
import argparse
from box import Box
import yaml
from helpers import *

"""
This is the file that is called to filter the SCADA data.
It takes the file path to the config_file as input.
"""


def filtering_scada_data(power_curve_df, df, config):
    """
    Main cleaning pipeline for SCADA data.

    Args:
        power_curve_df (pd.DataFrame): Reference power curve.
        df (pd.DataFrame): SCADA data.
        config (Box): Configuration object with thresholds and settings.

    Returns:
        pd.DataFrame
    """
    df = df.copy()
    n_turbines = config.n_turbines
    df = rename_turbine_columns(df)
    df = create_turbine_on_columns(df, n_turbines)
    for ti in range(n_turbines):
        print(f"Processing tubine {ti}...")
        df = adjust_power_if_turned_off(df,ti, config.limit_non_operational)
        df = filter_negative_power(df, ti)

    df = remove_if_many_turbines_off(df, config.n_turbines_off_limit)
    df = remove_curtailment_values(df)
    df = remove_outliers_fully_operational(power_curve_df, df, config.left_power_curve_bound,
                                           config.right_power_curve_bound, config.n_turbines)
    
    df = remove_nan_rows(df)
    df = remove_nan_rows(df, col_name='Curtailment mode_')

    df.reset_index(drop=True, inplace=True)
    df = add_cyclic_features(df, ['Nacelle position_', 'wd_'], n_turbines=n_turbines)
    df = add_cyclic_features(df, ['wd'])
    return df



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Filter SCADA data for each turbine.")
    parser.add_argument('--yaml_config', '-c', help="path to the yaml config file", type=str,required=False, default='processing_scada_data/config_filtering.yml')
    args = parser.parse_args()
    
    config = Box.from_yaml(filename=args.yaml_config, Loader=yaml.FullLoader)
    df = pd.read_parquet(config.file_path_scada)
    if config.use_weather_data:
        df_weather = pd.read_csv(config.file_path_weather)
        df = merge_global_local_features(df,df_weather)
        
    power_curve_df = pd.read_csv(config.power_curve_path)
    df = filtering_scada_data(power_curve_df,df,config)
    df = add_time_features(df,config)

    
    full_file_name = config.file_name_save

    print('Dataset size', len(df))
    df.to_parquet(full_file_name, engine="pyarrow")
    print('Train file saved successfully with name: ', full_file_name)