'''
Script that contains helper functions to filter the SCADA data
and to combine it with weather data with same time resolution.
'''
import numpy as np
import pandas as pd


def deg_to_cyclic(df, col_name, sin_name=None, cos_name=None):
    """
    Converts a degree-based feature into its sine and cosine components.

    Parameters:
        df (pd.DataFrame): Input dataframe.
        col_name (str): Name of the column with degree values.
        sin_name (str): Optional custom name for sine column.
        cos_name (str): Optional custom name for cosine column.

    Returns:
        df: Modified df
    """
    radians = np.deg2rad(df[col_name])
    sin_name = sin_name or f"{col_name}_sin"
    cos_name = cos_name or f"{col_name}_cos"
    df[sin_name] = np.sin(radians)
    df[cos_name] = np.cos(radians)
    return df

def add_cyclic_features(df, prefixes, n_turbines=None):
    """
    Adds cyclic (sine and cosine) features for angle-based columns.

    Parameters:
        df (pd.DataFrame): Input dataframe.
        prefixes (str or list): Prefix or list of prefixes for columns.
        n_turbines (int, optional): Number of turbines if turbine-specific.

    Returns:
        pd.DataFrame: DataFrame with additional cyclic features.
    """
    prefixes = [prefixes] if isinstance(prefixes, str) else prefixes
    for prefix in prefixes:
        if n_turbines is None:
            # Sinlge column
            if prefix in df.columns:
                df = deg_to_cyclic(df, prefix)
        else:
            for i in range(n_turbines):
                # For all turbines
                col = f"{prefix}{i:03d}"
                if col in df.columns:
                    df = deg_to_cyclic(df, col, f"{prefix.strip()}sin_{i:03d}", f"{prefix.strip()}cos_{i:03d}")
    return df

def rename_turbine_columns(df):
    """
    Renames turbine-related columns for consistency.

    Parameters:
        df (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: Renamed DataFrame.
    """
    df = df.copy()
    return df.rename(columns={
        col: col.replace("Wind speed_", "ws_")
                 .replace("Wind direction_", "wd_")
                 .replace("Power_", "pow_")
        for col in df.columns
        if col.startswith(("Wind speed_", "Wind direction_", "Power_"))
    })

def create_turbine_on_columns(df, n_turbines):
    """
    Initializes 'turbine_on' columns to 1 for each turbine.

    Parameters:
        df (pd.DataFrame): Input dataframe.
        n_turbines (int): Number of turbines.

    Returns:
        pd.DataFrame: DataFrame with new binary columns.
    """
    for i in range(n_turbines):
        df[f"turbine_on_{i:03d}"] = 1
    return df



def adjust_power_if_turned_off(df, ti, limit):
    """
    Sets power to zero if it falls in a range indicating the turbine was off.

    Parameters:
        df (pd.DataFrame): Input dataframe.
        ti (int): Turbine index.
        limit: Lower limit for how negative the power can be

    Returns:
        pd.DataFrame: Updated DataFrame with adjusted power.
    """
    pow_col = f"pow_{ti:03d}"
    on_col = f"turbine_on_{ti:03d}"
    mask = (df[pow_col] <= 0) & (df[pow_col] > limit)
    if mask.any():
        print(f"Turbine {ti:03d}: adjusting {mask.sum()} power values")
    df.loc[mask, [pow_col, on_col]] = 0 # sets both power and operational status to 0 if limit < power <= 0

    return df

def filter_negative_power(df, ti):
    """
    Filters out rows where turbine power is negative.

    Parameters:
        df (pd.DataFrame): Input dataframe.
        ti (int): Turbine index.

    Returns:
        pd.DataFrame: Updated DataFrame with NaN for negative values.
    """
    pow_col = f"pow_{ti:03d}"
    if pow_col in df.columns:
        mask = df[pow_col] < 0
        if mask.any():
            print(f"Turbine {ti:03d}: removing {mask.sum()} negative power values")
        df.loc[mask, pow_col] = np.nan
    return df



def remove_nan_rows(df, col_name='pow'):
    """
    Drop rows with NaN in any column starting with prefix.

    Args:
        df (pd.DataFrame): Input DataFrame.
        col_name (str): Column prefix (default 'pow').

    Returns:
        pd.DataFrame
    """
    columns = [col for col in df.columns if col.startswith(col_name)]
    df = df.dropna(subset=columns)
    df.reset_index(drop=True, inplace=True)
    return df



def remove_outliers_fully_operational(power_curve_df, df_in, offset_left=2, offset_right=2, num_turbines=16,
                                      wind_speed_col='Wind Speed [m/s]', power_col='Power Turbine [kW]'):
    """
    Remove data points outside wind-power envelope when turbine is operational.

    Args:
        power_curve_df (pd.DataFrame): Reference power curve.
        df_in (pd.DataFrame): SCADA data.
        offset_left (float): Lower offset from powercurve.
        offset_right (float): Upper offset from power curve.
        num_turbines (int): Number of turbines.

    Returns:
        pd.DataFrame
    """
    df = df_in.copy()
    ref_ws = power_curve_df[wind_speed_col][:35].values
    ref_power = power_curve_df[power_col][:35].values
    curtailment_cols = [col for col in df.columns if "curtailment" in col.lower()]

    for i in range(num_turbines):
        ws_col_turb = f"ws_{i:03d}"
        pow_col_turb = f"pow_{i:03d}"
        ws_turb = df[ws_col_turb].values
        pow_turb = df[pow_col_turb].values
        external_mask = ((df[curtailment_cols[i]] == 0.0) | (df[curtailment_cols[i]] == 18.0)).values # Mask al curtailments but 0 and 18
        keep_mask = np.ones(len(ws_turb), dtype=bool)
        special_mask = (pow_turb > 3500) | (pow_turb == 0) # Define special points (always kept) where power >3500 or power == 0.
        non_special_idx = np.where((external_mask) & (~special_mask))[0]

        if len(non_special_idx) > 0:
            expected_ws = np.interp(pow_turb[non_special_idx], ref_power, ref_ws) # Interpolate expected wind speed for non-special points using the reference curve.
            lower_env = expected_ws - offset_left
            upper_env = expected_ws + offset_right
            envelope_mask = (ws_turb[non_special_idx] >= lower_env) & (ws_turb[non_special_idx] <= upper_env) # Create an envelope mask: True if turbine's wind speed is within the envelope.
            keep_mask[non_special_idx] = envelope_mask

        # Update the original DataFrame: set power values that don't pass the combined mask to NaN.
        removed_count = np.sum(~keep_mask)
        print(f"Turbine {i:03d}: {removed_count} values removed as an outlier")
        df[pow_col_turb] = np.where(keep_mask, pow_turb, np.nan)

    return df


def remove_if_many_turbines_off(df, n_turbines_off_limit):
    """
    Drop rows where more than `n_turbines_off_limit` turbines are off.

    Args:
        df (pd.DataFrame): Input DataFrame.
        n_turbines_off_limit (int): Max number of turbines allowed to be off.

    Returns:
        pd.DataFrame
    """
    df = df.copy()
    turbine_columns = [col for col in df.columns if col.startswith('turbine_on_')]
    mask = (df[turbine_columns] == 0).sum(axis=1) > n_turbines_off_limit
    n_removed = mask.sum()
    print(f"{n_removed} rows removed because more than {n_turbines_off_limit} turbines were off")
    df = df[~mask]
    return df


def remove_curtailment_values(df):
    curtailment_values = [3.0, 19.0, 35.0, 36.0, 40.0, 43.0, 44.0, 45.0, 50.0, 51.0, 52.0]
    for col in df.columns:
        if col.startswith('Curtailment mode_'):
            before = df[col].notna().sum()
            df[col] = df[col].apply(lambda x: np.nan if x in curtailment_values else x)
            after = df[col].notna().sum()
            turbine_number = int(col.split('_')[-1]) + 1
            print(f"Turbine {turbine_number}: {before - after} values removed due to curtailment")
    return df

def merge_global_local_features(df_scada,df_weather):
 
    for df in [df_scada, df_weather]:
        if 'timestamp' in df.columns and 'time' not in df.columns:
            df.rename(columns={'timestamp': 'time'}, inplace = True)
        
        if 'date' in df.columns and 'time' not in df.columns:
            df.rename(columns={'date': 'time'}, inplace = True)

    if 'time' not in df_weather.columns:
        df_weather = df_weather.reset_index()

    df_weather['time'] = pd.to_datetime(df_weather['time'], utc = True)
    df_scada['time'] = pd.to_datetime(df_scada['time'], utc = True)

    df = df_scada.merge(df_weather, on='time', how='inner')
    return df


def add_time_features(df, config):
    df = df.copy()  
    
    if 'time' in df.columns:
        
        df['time'] = pd.to_datetime(df['time'], utc=True)
        
        if config.information_add.time_of_day:
            df['time_of_day'] = df['time'].dt.hour
            df['sin_time_of_day'] = np.sin(2 * np.pi * df['time_of_day'] / 24)
            df['cos_time_of_day'] = np.cos(2 * np.pi * df['time_of_day'] / 24)
        
        if config.information_add.month:
            # Extract month (1-12)
            df['month_of_year'] = df['time'].dt.month
            
            # Add cyclic features for month (1-12)
            df['sin_month_of_year'] = np.sin(2 * np.pi * df['month_of_year'] / 12)
            df['cos_month_of_year'] = np.cos(2 * np.pi * df['month_of_year'] / 12)
        
        else:
            print('The time feature is not yet implemented.')
    
    return df
