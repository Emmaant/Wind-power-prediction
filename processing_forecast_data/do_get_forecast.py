import os
import argparse
from datetime import date, timedelta
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
from datetime import datetime

from scipy.interpolate import LinearNDInterpolator

"""
Script for extracing forecast data 
To extract data specify coordinates in .csv file with columns 'Latitude', 'Longitude' and 
specify times in .parquet files with columns 'year', 'month', 'day', 'hour'

Output is saved in specified output directory
"""

def build_url(dat, run):
    """
    Function which builds url to download data from The Norwegian Meteorological institute
    """
    run_str = f"{int(run):02d}" 
    cutoff = date(2023, 10, 1) #Format of url was changed after this date
    if dat > cutoff:
        return f"https://thredds.met.no/thredds/dodsC/meps25epsarchive/{dat:%Y/%m/%d}/meps_det_ml_{dat:%Y%m%d}T{run_str}Z.ncml"
    else:
        return f"https://thredds.met.no/thredds/dodsC/meps25epsarchive/{dat:%Y/%m/%d}/meps_det_2_5km_{dat:%Y%m%d}T{run_str}Z.nc"

def process_forecast(ds, lat_farm, lon_farm):
    """
    Function which extracts u,v wind components at target height using grid points closest to turbine coordinates.
    Pressure and temperature are extracted at height level closest to weather_height
    Data is interpolated to turbine coordinates. 

    Returns:
       A DataFrame containing interpolated weather data for specified coordinates and forecast times. 
       The columns include:
        - forecast_time (datetime): Forecast timestamps repeated for each location.
        - latitude: Latitude of the farm or specified coordinates.
        - longitude: Longitude of the farm or specified coordinates.
        - u: wind component
        - v: wind component
        - wind_speed: Wind speed magnitude, calculated from u,v components.
        - wind_direction: Wind direction in degrees, calculated from u,v components.
        - surface_air_pressure: Interpolated surface air pressure (Pa).
        - temperature (float): Interpolated air temperature (K).
        - relative_humidity (float): Interpolated relative humidity (%).
    """
    ap = ds['ap'].isel(hybrid=slice(-nh, None)).values
    b = ds['b'].isel(hybrid=slice(-nh, None)).values

    weather_height = 2

    #Extract indecies closest to farm
    ix, iy = [], []
    for lataim, lonaim in zip(lat_farm, lon_farm):
        dist = np.sqrt((ds.longitude - lonaim) ** 2 + (ds.latitude - lataim) ** 2)
        flat_indices = dist.values.flatten().argsort()[:npo]
        ny, nx = ds.longitude.shape
        i, j = np.unravel_index(flat_indices, (ny, nx))
        ix.append(i)
        iy.append(j)

    #Get unique index
    tuples = [tuple(pair) for i, j in zip(ix, iy) for pair in zip(i,j)]
    unique_tuples = list(set(tuples))
    y_idx = [idx[0] for idx in unique_tuples]
    x_idx = [idx[1] for idx in unique_tuples]

    #Initialize pressure, temp, u, v wind arrays for closest coordinates
    ps = np.zeros((nf, 1, len(y_idx)))
    ta = np.zeros((nf, nh, len(y_idx)))
    u = np.zeros((nf, nh, len(y_idx)))
    v = np.zeros((nf, nh, len(y_idx)))

    for ip in range(len(x_idx)):
        ps[:, :, ip] = ds['surface_air_pressure'].isel(time=slice(0, nf), y=y_idx[ip], x=x_idx[ip]).values
        ta[:, :, ip] = ds['air_temperature_ml'].isel(time=slice(0, nf), hybrid=slice(-nh, None), y=y_idx[ip], x=x_idx[ip]).values
        u[:, :, ip] = ds['x_wind_ml'].isel(time=slice(0, nf), hybrid=slice(-nh, None), y=y_idx[ip], x=x_idx[ip]).values
        v[:, :, ip] = ds['y_wind_ml'].isel(time=slice(0, nf), hybrid=slice(-nh, None), y=y_idx[ip], x=x_idx[ip]).values

    p = np.repeat(ap, len(x_idx) * nf).reshape((nh, nf, len(x_idx))) + np.outer(b, ps.flatten()).reshape((nh, nf, len(x_idx)))
    p = np.moveaxis(p, 0, -2)
    pt = np.concatenate((p, ps), axis=1)

    dz = 287 * ta / 9.81 * np.log(pt[:, 1:, :] / pt[:, :-1, :])
    zv = np.cumsum(dz[:, ::-1, :], axis=1)[:, ::-1, :]

    #Interpolate to specified height
    u_interp, v_interp = np.full((nf, len(x_idx)), np.nan), np.full((nf, len(x_idx)), np.nan)
    ps_value, ta_value, rh_value = np.full((nf, len(x_idx)), np.nan), np.full((nf, len(x_idx)), np.nan), np.full((nf, len(x_idx)), np.nan)
    for t in range(nf):
        for ip in range(len(x_idx)):
            h_profile = zv[t, :, ip]
            u_profile = u[t, :, ip]
            v_profile = v[t, :, ip]

            ih = np.abs(h_profile - weather_height).argmin() #Take closest index for temperature, unsure if interpolation makes sense as it it unclear if there exsists two points around target height

            ps_value[t, ip] = ps[t,0,ip]
            ta_value[t, ip] = ta[t,ih, ip]

            sh = ds['specific_humidity_ml'].isel(time=t, hybrid=ih, y=y_idx[ip], x=x_idx[ip]).values
            rh_value[t, ip] = 0.263 * sh * ps[t,0, ip] * 1/np.exp(17.67*(ta[t,ih, ip] - 273.16)/(ta[t,ih, ip] - 29.65))

            if np.any(h_profile <= target_height) and np.any(h_profile >= target_height):
                idx = np.argsort(h_profile)
                u_interp[t, ip] = np.interp(target_height, h_profile[idx], u_profile[idx])
                v_interp[t, ip] = np.interp(target_height, h_profile[idx], v_profile[idx])


    #Interpolate to farm coordinates
    coords = np.array([[ds.latitude.values[y, x], ds.longitude.values[y, x]] for y, x in unique_tuples])
    new_coords = np.array((lat_farm, lon_farm)).T
    u_interp_new = np.full((nf, len(new_coords)), np.nan)
    v_interp_new = np.full((nf, len(new_coords)), np.nan)
    ps_interp = np.full((nf, len(new_coords)), np.nan)
    ta_interp = np.full((nf, len(new_coords)), np.nan)
    rh_interp = np.full((nf, len(new_coords)), np.nan)

    for t in range(nf):
        # Interpolators from model grid to farm points
        interpolate_u = LinearNDInterpolator(coords, u_interp[t, :])
        interpolate_v = LinearNDInterpolator(coords, v_interp[t, :])
        interpolate_ps = LinearNDInterpolator(coords, ps_value[t, :])
        interpolate_ta = LinearNDInterpolator(coords, ta_value[t, :])
        interpolate_rh = LinearNDInterpolator(coords, rh_value[t, :])

        u_interp_new[t, :] = interpolate_u(new_coords)
        v_interp_new[t, :] = interpolate_v(new_coords)
        ps_interp[t, :] = interpolate_ps(new_coords)
        ta_interp[t, :] = interpolate_ta(new_coords)
        rh_interp[t, :] = interpolate_rh(new_coords)

    wind_speed = np.sqrt(u_interp_new**2 + v_interp_new**2)
    wind_dir = (270 - np.degrees(np.arctan2(v_interp_new, u_interp_new))) % 360

    forecast_times = ds.time.values[:nf]

    df = pd.DataFrame({
        "forecast_time": np.repeat(forecast_times, len(new_coords)),
        "latitude": np.tile(lat_farm, nf),
        "longitude": np.tile(lon_farm, nf),
        "u": u_interp_new.flatten(),
        "v": v_interp_new.flatten(),
        "wind_speed": wind_speed.flatten(),
        "wind_direction": wind_dir.flatten(),
        "surface_air_pressure": ps_interp.flatten(),
        "temperature": ta_interp.flatten(),
        "relative_humidity": rh_interp.flatten()
    })

    return df

def process_single_file(date, lat_farm, lon_farm,run):
    """
    Processes a single weather forecast file for a specific date and run time.
    Parameters:
        date (datetime.date): The date of the forecast.
        lat_farm : Array of latitudes for interpolation.
        lon_farm : Array of longitudes for interpolation.
        run (int): Forecast run hour (e.g., 0, 6, 12, 18 UTC).
    """
    out_file = f"{output_dir}/wind_forecast_{date:%Y%m%d}T{run}Z.parquet"
    if os.path.exists(out_file):
        print(f"Skipped {out_file} (already exists)")
        return
    url = build_url(date, run)
    try:
        ds = xr.open_dataset(url)
        df = process_forecast(ds, lat_farm, lon_farm)
        df["forecast_run"] = f"{date:%Y-%m-%d}T{run}Z"
        df.to_parquet(out_file, index=False)
    except Exception as e:
        print(f"Failed to process {url}: {e}")

def process_file(df_time):
    """
    Iterates through a DataFrame of forecast timestamps and processes each corresponding forecast run.

    Parameters:
        df_time: DataFrame with columns 'year', 'month', 'day', and 'hour', representing forecast initialization times.
    """
    for row in tqdm(df_time.itertuples(index=False), total=len(df_time)):
        d = date(row.year, row.month, row.day)
        run = row.hour
        process_single_file(d, lat_farm, long_farm, run)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process MEPS wind forecast data")
    parser.add_argument('--file', '-f', type=str, default=None, help="Specify file path with times to extract")
    parser.add_argument('--coordinates', '-c', type=str, default=None, help="Specify file path to .csv wind farm coordinates, should contain columns 'Latitude', 'Longitude'")
    parser.add_argument('--output_dir', '-o', type=str, default=None, help="Specify path to output directory")

    args = parser.parse_args()

    target_height = 117
    forecast_runs = ['00', '06', '12', '18']
    nh = 7  # number of vertical levels
    nf = 24  # Extract 24 forecast hours
    npo = 3  # number of nearby grid points to extract
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    #File to wind farm coordinates
    windFarmCoord = pd.read_csv(args.coordinates)
    lat_farm, long_farm = windFarmCoord[['Latitude', 'Longitude']].to_numpy().T

    df_time = pd.read_parquet(args.file)
    process_file(df_time)

