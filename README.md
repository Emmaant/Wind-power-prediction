# Master Thesis: Machine learning for Wind Power Prediction

This repository contains scripts for training machine learning models in wind power prediction. The work was done as a master thesis project at Chalmers University of Technology together with RISE. The thesis presents a comparative study of machine learning (ML) models and the deep learning (DL) model graph neural network (GNN) for wind power prediction and short- to medium-term forecasting in wind farms. Using high-resolution SCADA
data from a 16-turbine onshore wind farm in Sweden, along with re-analysis and forecast weather datasets, various models including Random Forest (RF), eXtremeGradient Boosting (XGBoost), k-nearest neighbours (kNN), Multi-Layer Perceptron (MLP), and GNNs were trained and evaluated. Two baseline approaches, the farm’s
theoretical power curve and FLORIS wake model, were used as references. Results
show that ML models outperform baseline models in predicting wind power output,
with GNN achieving the best overall performance, although all ML models perform
similarly. However, the findings indicate that re-analysis data with low spatial
resolution fails to adequately capture local weather conditions necessary for accurate
power prediction. The study also investigates the effects of input feature selection,
temporal resolution, and multi-task learning on model performance. Furthermore,
it identifies challenges related to input data quality, particularly in the estimation of
global wind conditions from SCADA-based measurements. These results underscore
the potential of ML methods for wind power applications and highlight the critical
importance of accurately representing global weather data, as well as accounting for
discrepancies between training data and forecast data.


## Repository structure

```yaml
data/

gnn_framework/
├── processor_settings/
│   └── optimizer.py
├── config.yml
├── gnn_architecture.py
└── train_gnn.py

ml_framework/
├── config_files/
│   ├── feature_ws_wd_turbineon/
│   │   ├── config.yml
│   └── ...
├── config_files_mlp/
│   ├── multi_feature_ws_wd_turbineon/
│   └── ├── config.yml
├── do_inference.py
├── do_inference_forecast.py
├── do_inference_forecast_mlp.py 
├── do_inference_mlp.py
├── do_train_ml_model.py
└── do_train_mlp_model.py

processing_forecast_data/
├── do_get_forecast.py
└── do_reformat_forecast_data.py
processing_scada_data/
├── config_filtering.yml
├── do_filtering.py
└── helpers.py
processing_weather_data/


```

## How to create environment required to run code

```console  
git clone https://github.com/Emmaant/Wind-power-prediction
```

Navigate to the cloned directory

```console
cd Wind-power-prediction
```

Create environment

```console
conda create --name <env> --file requirements.txt
```

## How to filter SCADA data
To preprocess the raw SCADA data:

1. Open the configuration file:  
   `processing_scada_data/config_filtering.yml`

2. Set the path to the **unfiltered SCADA data**.

3. *(Optional)* If you want to include external weather data (e.g., reanalysis data), specify its path in the same config file.  
    **Note**: The weather data must have the **same time resolution** as the SCADA data.

To run the filtering process:

```bash
python processing_scada_data/do_filtering.py
```

## How to create graphs
Before training the GNN model, graph data must be generated from raw input. This is done using the script:
```bash
python process_train_data/do_create_graph.py -config_path 'generate_data/graph_config.yml'
```
To generate graphs, you need to provide a configuration file. A sample config is available in generate_data/graph_config.yml

##### The config file allows you to specify:
- To use simulated data or not
- Path to the raw data file.
- Node features:
   - Local features: Specific to each turbine (identified by _XXX).
   - Global features: Shared across all nodes in the graph.
- Wake settings, which defines how graph edges (connections between turbines) are constructed, based on wake modeling.

Generated graphs are saved under:
```bash
Data/graphs/
```

## How to train GNN
To train the GNN model:

1. Open the GNN configuration file:
   `gnn_framework/config.yml`

2. Set the path to the folder containing the **graph data**.

To train the GNN model, run:
```bash
python gnn_framework/do_train_gnn_model.py
```

## How to train ML models (NOT Finalized)
To train ML models the code in ml_framework is used. To train MLP use do_train_mlp_model.py or sklearn models can be trained do_train_ml_model.py. Tabular data is used as input.

### How to train sklearn models
To train sklearn models a directory of subdirectories with config files is needed. Each config file specifies the features to use during training. Local features are specific for each turbine and all features ending in _000 - _016 are added. Global features are simply added as specified. In the config file one needs to specify if one model for each turbine is to be trained or one model for all turbines. The trained models are saved in the subdirectories. 

The models trained are kNN regressor, XGBoost regressor and Random Forest Regressors. Others could be added to the dictionary `regressors` if needed. The hyperparameters are tuned using grid search and specified in the dictionary  `param_grids`. To modify, modify script directly.

To train the models run:

```bash
python ml_framework/do_train_ml_model.py -c 'ml_framework/config_files'
```

Only the best models for each subdirectory is saved. Training is done on CPU. Training on GPU is not supported by the script.

### How to train MLP models
To train MLP model a directory of subdirectories with config files is needed. Each config files specifies the features to use during training.  Local features are specific for each turbine and all features ending in _000 - _016 are added. Global features are simply added as specified. One model is trained for predicting the power output for all turbines.

To train models run:

```bash
python ml_framework/do_train_mlp_model.py -c 'ml_framework/config_files_mlp'
```

Optuna is used for hyperparameter tuning. To specify number of optuna trails to run use -n. Only the best models for each subdirectory is saved. Training can be done on GPU or CPU.

 **Note**: If you want to add local statistical features from graph, in degree or out degree these can be created by using script: 

```bash
python generate_data/do_create_degree_features.py'
```

see section See [Create Degree Features](##create-degree-features) below.

## Create Degree Features
Statistical properties can be calculated from a graph. Adding these when training ML models which take tabular data as input could improve training as it for example can give the models information about what turbines are in wake or upstream. 

Therefore, a script for calculating the in degree and out degree from a set of graphs was created. To run script:

```bash
python generate_data/do_create_degree_features.py' --yaml_config 'generate_data/data_config.yml' 
```
The config file specifies the path to the data file to add features to and the wake settings to use when creating features.
Columns added are out_degree_(angle, base_width, length)_XXX, in_degree_(angle, base_width, length)_XXX where _XXX denotes the turbine index.

The new dataframe is saved in .parquet format and by default in folder 'data/train_data/ml_data/'. To change output directory use '--output_dir' or '-o'.

## How to get metno forcast data
To obtain forecast data from MET Norway a script has been created. To extract data specify coordinates in .csv file with columns 'Latitude', 'Longitude' and specify times in .parquet files with columns 'year', 'month', 'day', 'hour'

```bash
python processing_forecast_data/do_get_forecast.py -f 'processing_forecast_data/df_times.parquet' -c 'data/coordinates_wind_farm.csv' -o 'data/forecast'
```

The forecasts are saved in .parquet format in the specified output directory. To change output directory use '--output_dir' or '-o'.

In the script you can specify the target_height i.e. what height to interpolate the win dspeed and wind direction to.  The number of forecast hours to extract. The number of nearby gridpoints in the forecast data to use for the coordinates you want to extract.

The scripts provides interpolated weather data for specified coordinates and forecast times. 
The columns include:
   - forecast_run (datetime): The time the forecast was made.
   - forecast_time (datetime): The time of the forecast.
   - latitude: Latitude of the farm or specified coordinates.
   - longitude: Longitude of the farm or specified coordinates.
   - u: wind component
   - v: wind component
   - wind_speed: Wind speed magnitude, calculated from u,v components.
   - wind_direction: Wind direction in degrees, calculated from u,v components.
   - surface_air_pressure: Interpolated surface air pressure (Pa).
   - temperature (float): Interpolated air temperature (K).
   - relative_humidity (float): Interpolated relative humidity (%).

Run:

```bash
python processing_forecast_data/do_reformat_forecast_data.py --forecast_dir 'data/forecast' --scada_data  --index_path
```
To merge forecast data with SCADA data. You will need to specify forecast directory, path to SCADA data and index to use.

