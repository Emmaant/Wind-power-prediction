"""
Script for training machine learning models. 
Regressors are specified in script together with hypermarameters to tune
using grid search. To change modify script.

Configure training by providing directory of config file(s) in .yml format. 
Example file is availible in directory config_files
"""

import argparse
import wandb
import os
from itertools import product
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from box import Box
import yaml
from pathlib import Path
import joblib
import shutil

from utils.helpers_ml_models import *

# --- Model Class ---
class Model:
    def __init__(self, regressor, params):
        self.regressor = regressor
        self.params = params
        self.multi_task = bool
    
    def __str__(self):
        return f"Regressor: {self.regressor}, Parameters: {self.params}"

    def fit(self, data, df_tr, df_val):
        x_tr = df_tr[data.x_columns].values
        y_tr = df_tr[data.y_columns].values
        x_val = df_val[data.x_columns].values
        y_val = df_val[data.y_columns].values

        if not self.multi_task:
            y_tr = y_tr.ravel()
            y_val = y_val.ravel()

        self.regressor.set_params(**self.params)
        self.regressor.fit(x_tr, y_tr)

        y_pred = self.regressor.predict(x_val)

        if self.multi_task:
            y_pred.reshape(-1, 16)

        mse = np.mean((y_pred - y_val) ** 2, axis=0)
        
        return mse, self.regressor

# --- Trainer Class ---
class Trainer:
    def __init__(self, config, regressors, param_grids,subdir):
        self.config = config
        self.regressors = regressors
        self.param_grids = param_grids
        self.data = Data(config)
        self.data.get_data_columns()
        self.df_tr, self.df_val = self.data.get_data()
        self.dir_path = subdir

    def train(self):
        wandb.login()

        for method, regressor in self.regressors.items():
            param_names = list(self.param_grids[method].keys())
            param_combinations = list(product(*self.param_grids[method].values()))

            best_mse = np.ones(config.data_settings.num_turbines) * np.inf
            best_model_dir = self.dir_path / f"best_model_{method}"
            os.makedirs(best_model_dir, exist_ok=True)

            best_model = {'mse': np.ones(config.data_settings.num_turbines),'run_name': ''}

            for param_set in param_combinations:
                params = dict(zip(param_names, param_set))

                run_name = self._generate_run_name(method, params)
                model = Model(regressor, params)
                model.multi_task = self.config.data_settings.multitask

                model_dir = self.dir_path / f"model"
                os.makedirs(model_dir, exist_ok=True)

                if self.config.data_settings.multitask:
                    mse, model_results = model.fit(self.data, self.df_tr, self.df_val)
                    joblib.dump(model_results, os.path.join(model_dir, "multitask_model.pkl"))

                    if mse.sum() < best_mse.sum():
                        best_model['mse'] = mse
                        best_model['run_name'] = run_name

                        shutil.rmtree(best_model_dir)
                        shutil.copytree(model_dir, best_model_dir)
                        shutil.rmtree(model_dir)
                    else:
                        shutil.rmtree(model_dir)

                else:
                    mse = np.zeros(16)
                    for i in range(16):
                        self.data.get_column_names_training(i)
                        mse_value, model_result = model.fit(self.data, self.df_tr, self.df_val)
                        joblib.dump(model_result, os.path.join(model_dir, f"model_turbine_{i}.pkl"))
                        mse[i] = mse_value

                    if mse.sum() < best_mse.sum():
                        best_model['mse'] = mse
                        best_model['run_name'] = run_name

                        shutil.rmtree(best_model_dir)
                        shutil.copytree(model_dir, best_model_dir)
                        shutil.rmtree(model_dir)
                    else:
                        shutil.rmtree(model_dir)

            self._log_results(best_model['run_name'], best_model['mse'])

        return best_model

    def _generate_run_name(self, method, params):
        return f"method_{method}_multitask_{self.config.data_settings.multitask}_local_" \
               f"{self.data.local_features}_global_{self.data.global_features}_params{params}"

    def _log_results(self, run_name, mse):
        run = wandb.init(project="ml_models_hourly", name=run_name, reinit=True)
        for mse_value in mse:
            wandb.log({f"MSE": mse_value})
        run.finish()

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description="Train XGBOOST regressor")
    parser.add_argument('--yaml_config', '-c', help="path to the directory with yaml config files", type=str,
                        required=False, default=None)

    args = parser.parse_args()

    #Specify parameters and values for tuning using grid search
    param_grids = {
    'XGBoost': {
        'max_depth': [3,5],
        'learning_rate': [0.1, 0.05],
        'n_estimators': [100, 200]
    },
    'RF': {
        'n_estimators': [100,200],
        'max_depth': [5,10],
        'min_samples_split': [5]
    },
    'kNN': {
        'n_neighbors': [2, 5,10, 20, 50,100, 200]
    }
}
    config_dir = Path(args.yaml_config)

    for subdir in config_dir.iterdir():
        print('Running directory', subdir)
        if subdir.is_dir():
            print(subdir)
            yaml_files = list(subdir.glob("*.yml"))
            if yaml_files:
                config = Box.from_yaml(filename=yaml_files[0], Loader=yaml.FullLoader)

            if config.data_settings.multitask:
                regressors = {
                    'kNN': KNeighborsRegressor(),
                    'XGBoost': xgb.XGBRegressor(n_jobs=1,tree_method="hist", objective='reg:squarederror', multi_strategy="multi_output_tree", subsample=0.6),
                    'RF': RandomForestRegressor(),
                }
            else:
                regressors = {
                    'kNN': KNeighborsRegressor(),
                    'XGBoost': xgb.XGBRegressor(n_jobs=1,tree_method="hist", objective='reg:squarederror', subsample=0.6),
                    'RF': RandomForestRegressor(),
                }
            
            trainer = Trainer(config, regressors, param_grids,subdir)
            best_model = trainer.train()
