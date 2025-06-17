"""
Script for running inference using forecast data on machine learning models.
"""

import argparse
from box import Box
import yaml
import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd
from pathlib import Path
import joblib

from utils.helpers_ml_models import Data 

class Inference:
    def __init__(self, config, model_dir):
        self.config = config
        self.data = Data(config)
        self.data.get_data_columns()
        self.df_tr, self.df_val = self.data.get_data()
        self.model_dir = Path(model_dir)

    def run(self):
        model_names = ['best_model_KNN', 'best_model_RF', 'best_model_XGBOOST']

        base_model_path = self.model_dir

        for model_name in model_names:
            self.model_dir = Path(base_model_path) / model_name # Adjust base_model_path accordingly
            self._evaluate_model(self.model_dir)

    def _evaluate_model(self, model_dir):

        df_forecast = pd.read_parquet('data/scada_ifs_forecast_test.parquet')

        num_turbines = self.config.data_settings.num_turbines
        y_pred_org = np.zeros((len(df_forecast), num_turbines))
        y_pred_for_org = np.zeros((len(df_forecast), num_turbines))
        y_test_org = np.zeros_like(y_pred_org)

        org_columns = ['ws','wd_cos', 'wd_sin','sin_time_of_day','cos_time_of_day','sin_month_of_year','cos_month_of_year']
        forcast_columns = ['ws_for_avg','wd_for_cos_avg', 'wd_for_sin_avg','sin_time_of_day','cos_time_of_day','sin_month_of_year','cos_month_of_year']

        scale_index = self.data.scale_cols.index('ws')

        df_forecast['ws'] = (df_forecast['ws'] - self.data.scaler.data_min_[scale_index]) /  (self.data.scaler.data_max_[scale_index] - self.data.scaler.data_min_[scale_index])
        df_forecast[['wd_cos', 'wd_sin']] = (df_forecast[['wd_cos', 'wd_sin']] + 1) / 2

        df_forecast['ws_for_avg'] = (df_forecast['ws_for_avg'] - self.data.scaler.data_min_[scale_index]) /  (self.data.scaler.data_max_[scale_index] - self.data.scaler.data_min_[scale_index])
        df_forecast[['wd_for_cos_avg', 'wd_for_sin_avg','sin_time_of_day','cos_time_of_day','sin_month_of_year','cos_month_of_year']] = (df_forecast[['wd_for_cos_avg', 'wd_for_sin_avg','sin_time_of_day','cos_time_of_day','sin_month_of_year','cos_month_of_year']] + 1) / 2

        if self.config.data_settings.multitask:
            model = joblib.load(model_dir / 'multitask_model.pkl')

            y_test = df_forecast.filter(regex=r'^pow_0(0[0-9]|1[0-6])$').to_numpy()
            
            x_test_for = pd.concat([
                df_forecast.filter(regex='turbine_on_'),
                df_forecast[forcast_columns]
            ], axis=1).to_numpy()

            x_test = pd.concat([
                df_forecast.filter(regex='turbine_on_'),
                df_forecast[org_columns]
            ], axis=1).to_numpy()


            y_pred_for = model.predict(x_test_for)
            y_pred = model.predict(x_test)

            col_indices = [self.data.scale_cols.index(col) for col in self.data.y_columns]
            y_pred_for_org = y_pred_for * (self.data.scaler.data_max_[col_indices] - self.data.scaler.data_min_[col_indices]) + self.data.scaler.data_min_[col_indices]
            y_pred_org = y_pred * (self.data.scaler.data_max_[col_indices] - self.data.scaler.data_min_[col_indices]) + self.data.scaler.data_min_[col_indices]
            y_test_org = y_test

        else:
            for i in range(num_turbines):
                ti = f"{i:03d}"
                model = joblib.load(model_dir / f'model_turbine_{i}.pkl')

                self.data.get_column_names_training(i)
                
                x_test_for_df = pd.concat([
                    df_forecast[f'turbine_on_{ti}'],
                    df_forecast[forcast_columns],
                ], axis=1)

                x_test_for = x_test_for_df.to_numpy()

                x_test_df= pd.concat([
                    df_forecast[f'turbine_on_{ti}'],
                    df_forecast[org_columns]
                ], axis=1)

                x_test = x_test_df.to_numpy()

                y_test = df_forecast[f'pow_{ti}']

                col_indices = [self.data.scale_cols.index(f'pow_{ti}')]
                y_pred_for = model.predict(x_test_for)
                y_pred = model.predict(x_test)

                y_pred_for_org[:, i] = y_pred_for * (self.data.scaler.data_max_[col_indices] - self.data.scaler.data_min_[col_indices]) + self.data.scaler.data_min_[col_indices]
                y_pred_org[:, i] = y_pred * (self.data.scaler.data_max_[col_indices] - self.data.scaler.data_min_[col_indices]) + self.data.scaler.data_min_[col_indices]
                y_test_org[:, i] = y_test

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
            ax.scatter(y_test_org[:, i], y_pred_org[:, i], s=0.5)
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
            ax.scatter(y_test_org[:, i], y_pred_for_org[:, i], s=0.5)
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
    print(config_dir)
    yaml_files = list(config_dir.glob("*.yml"))
    print(yaml_files)
    config = Box.from_yaml(filename=yaml_files[0], Loader=yaml.FullLoader)

    global_feature_list = ['ws','wd_cos','wd_sin','sin_time_of_day','cos_time_of_day','sin_month_of_year','cos_month_of_year']

    if config.data_settings.global_feature_list == global_feature_list:

        inference_runner = Inference(config, config_dir)
        inference_runner.run()
        
    else:
        print('Only ws, wd_cos and wd_sin are supported features for forecasting')