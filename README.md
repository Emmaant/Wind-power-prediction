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
├── gnn_framework/ \
    ├── processor_settings \
    ├── config.yml \
    ├── gnn_architecture.py \
    ├── train_gnn.py \
├── ml_framework/ \
    ├── config_files/ \
    ├── config_files_mlp/ \
    ├── do_inference_forecast_mlp.py \
    ├── do_inference_forecast.py \
    ├── do_inference_mlp.py \
    ├── do_inference.py \
    ├── do_train_ml_model.py \
    ├── do_train_mlp_model.py \
├── processing_forecast_data/ \
├── processing_scada/ \
├── processing_weather_data/\


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


## How to train GNN




## How to train ML models




## How to get metno forcast data






