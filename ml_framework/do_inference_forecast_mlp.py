"""
Script for running inference using forecast data on mlp models.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from box import Box
from utils.helpers_ml_models import Data 

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.0):
        super(MLP, self).__init__()
        layers = []

        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dim, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class Inference:
    def __init__(self, config, model_dir):
        self.config = config
        self.data = Data(config)
        self.data.get_data_columns()
        self.df_tr, self.df_val = self.data.get_data()
        
        self.model_dir = Path(model_dir)
        self.target_cols = self.data.target_columns
        self.input_cols = self.data.data_columns
        self.device = torch.device("cpu")  # force CPU

    def run(self):
        model_names = ['best_MLP']

        base_model_path = self.model_dir

        for model_name in model_names:
            self.model_dir = Path(base_model_path) / model_name # Adjust base_model_path accordingly
            self._evaluate_model(self.model_dir)

    def _evaluate_model(self, model_dir):

        num_turbines = 16

        df_forecast = pd.read_parquet('data/scada_ifs_forecast_test.parquet')

        pt_files = list(model_dir.glob("*.pt"))
        pt_files = [f for f in pt_files if not f.name.startswith('.')]
        if len(pt_files) == 0:
            print(f"No .pt model file found in: {model_dir}")
            return
        elif len(pt_files) > 1:
            print(f"Multiple .pt files found in: {model_dir}, using the first one: {pt_files[0].name}")

        model_path = pt_files[0]
        print(f"Using model file: {model_path.name}")

        local_columns = [f'in_degree_(20, 2, 2720)_{i:03d}' for i in range(16)]
        org_columns = ['ws','wd_cos', 'wd_sin'] + local_columns
        forcast_columns = ['ws_for_avg','wd_for_cos_avg', 'wd_for_sin_avg'] + local_columns

        scale_index = self.data.scale_cols.index('ws')
        df_forecast['ws'] = (df_forecast['ws'] - self.data.scaler.data_min_[scale_index]) /  (self.data.scaler.data_max_[scale_index] - self.data.scaler.data_min_[scale_index])
        scale_index = self.data.scale_cols.index(local_columns)
        df_forecast[local_columns] = (df_forecast[local_columns] - self.data.scaler.data_min_[scale_index]) /  (self.data.scaler.data_max_[scale_index] - self.data.scaler.data_min_[scale_index])

        df_forecast.filter(regex=r'^pow_0(0[0-9]|1[0-6])$')

        df_forecast[['wd_cos', 'wd_sin']] = (df_forecast[['wd_cos', 'wd_sin']] + 1) / 2

        df_forecast['ws_for_avg'] = (df_forecast['ws_for_avg'] - self.data.scaler.data_min_[scale_index]) /  (self.data.scaler.data_max_[scale_index] - self.data.scaler.data_min_[scale_index])
        df_forecast[['wd_for_cos_avg', 'wd_for_sin_avg']] = (df_forecast[['wd_for_cos_avg', 'wd_for_sin_avg']] + 1) / 2

        x_test_for = torch.tensor(pd.concat([
                df_forecast.filter(regex='turbine_on_'),
                df_forecast[forcast_columns]
            ], axis=1).values, dtype=torch.float32)

        x_test = torch.tensor(pd.concat([
                df_forecast.filter(regex='turbine_on_'),
                df_forecast[org_columns]
            ], axis=1).values, dtype=torch.float32)

        y_test = torch.tensor(df_forecast.filter(regex=r'^pow_0(0[0-9]|1[0-6])$').values, dtype=torch.float32)
        
        model = MLP(
            input_dim= len(self.input_cols),
            hidden_dim=self.config.hidden_dim,
            output_dim= len(self.target_cols),
            num_layers=self.config.num_layers,
        )
        
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        with torch.no_grad():
            y_pred = model(x_test.to(self.device)).numpy()
            y_pred_for = model(x_test_for.to(self.device)).numpy()

        y_test_org = y_test.numpy()

        col_indices = [self.data.scale_cols.index(f'pow_{ti:03d}') for ti in range(num_turbines)]

        # Rescale
        y_pred_org = y_pred * (self.data.scaler.data_max_[col_indices] - self.data.scaler.data_min_[col_indices]) + self.data.scaler.data_min_[col_indices]
        y_pred_for_org = y_pred_for * (self.data.scaler.data_max_[col_indices] - self.data.scaler.data_min_[col_indices]) + self.data.scaler.data_min_[col_indices]

        result_dict = self.get_error(y_test_org, y_pred_org)
        result_dict_for = self.get_error(y_test_org, y_pred_for_org)

        print(f"RMSE per turbine: {result_dict["rmse_per_turbine"]}")
        print(f"RMSE per turbine forcast: {result_dict_for["rmse_per_turbine"]}")


        # --- Saving results ---
        if config.data_settings.hourly_data:
            save_path = model_dir / "inference_plot_test_hourly.png"
            save_path_forcast = model_dir / "inference_plot_test_hourly_forecast.png"
            file_path_save = model_dir / "results_test_hourly.json"
            file_path_save_forcast = model_dir / "results_test_hourly_forcast.json"
        else:
            save_path = model_dir / "inference_plot_test.png"
            save_path_forcast = model_dir / "inference_plot_test_forecast.png"
            file_path_save = model_dir / "results_test.json"
            file_path_save_forcast = model_dir / "results_test_forcast.json"

        with open(file_path_save, "w") as f:
            json.dump(result_dict, f, indent=4)
        print(f"Saved .json file with results {file_path_save}")

        with open(file_path_save_forcast, "w") as f:
            json.dump(result_dict_for, f, indent=4)
        print(f"Saved .json file with results {file_path_save_forcast}")

        # --- Plotting ---
        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        axes = axes.flatten()
        for i in range(num_turbines):
            ax = axes[i]
            ax.scatter(y_pred_org[:, i], y_test_org[:, i], s=0.5)
            ax.set_title(f'Turbine {i:02d} | RMSE: {result_dict["rmse_per_turbine"][i]:.2f}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')

        plt.tight_layout()
        fig.suptitle(f'Avg RMSE per turbine: {np.mean(result_dict["rmse_per_turbine"]):.2f}', fontsize=16, y=1.02)

        print(save_path)
        fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved inference plot to {save_path}")


        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        axes = axes.flatten()
        for i in range(num_turbines):
            ax = axes[i]
            ax.scatter(y_pred_for_org[:, i], y_test_org[:, i], s=0.5)
            ax.set_title(f'Turbine {i:02d} | RMSE: {result_dict_for["rmse_per_turbine"][i]:.2f}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')

        plt.tight_layout()
        fig.suptitle(f'Avg RMSE per turbine: {np.mean(result_dict_for["rmse_per_turbine"]):.2f}', fontsize=16, y=1.02)

        print(save_path_forcast)
        fig.savefig(save_path_forcast, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved inference plot to {save_path_forcast}")

    def get_error(self,y_org, y_pred):
        mask = y_org != 0
        error = (y_pred - y_org) * mask

        mse_per_row = (error ** 2).mean(axis=1)
        mae_per_row = np.abs(error).mean(axis=1)
        rmse_per_turbine = np.sqrt(np.mean(error**2, axis=0))

        total_true = (y_pred * mask).sum(axis=1)
        total_pred = (y_org * mask).sum(axis=1)
        farm_error = total_true - total_pred
        farm_rmse = np.sqrt((farm_error ** 2).mean())
        farm_mae = np.abs(farm_error).mean()

        return {
            "mse_per_row": mse_per_row.tolist(), 
            "mae_per_row": mae_per_row.tolist(),
            "rmse_per_turbine": rmse_per_turbine.tolist(),
            "farm_rmse": float(farm_rmse),
            "farm_mae": float(farm_mae),
        }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run inference on saved models")
    parser.add_argument('--model_dir', '-m', type=str,
                        help="Path to directory containing models .pkl files", default=None)

    args = parser.parse_args()

    config_dir = Path(args.model_dir)
    yaml_files = list(config_dir.glob("config.yml"))
    print(yaml_files)
    config = Box.from_yaml(filename=yaml_files[0], Loader=yaml.FullLoader)

    global_feature_list = ['ws','wd_cos','wd_sin']

    print(config.data_settings.global_feature_list)

    if config.data_settings.global_feature_list == global_feature_list:

        inference_runner = Inference(config, config_dir)
        inference_runner.run()
        
    else:
        print('Only ws, wd_cos and wd_sin are supported features for forecasting')
