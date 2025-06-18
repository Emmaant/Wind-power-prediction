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


## Repository structure (NOT Finalized)

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
python generate_data/do_create_graph.py -config_path 'generate_data/graph_config.yml'
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



## How to train ML models




## How to get metno forcast data






