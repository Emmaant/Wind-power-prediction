"""
Script for running inference on validation data for machine learning models
"""

import argparse
from box import Box
import yaml
import matplotlib.pyplot as plt
import json
import numpy as np
from pathlib import Path
import joblib

from utils.helpers_ml_models import *

class Inference:
    def __init__(self, config, model_dir):
        self.config = config
        self.data = Data(config)
        self.data.get_data_columns()
        self.df_tr, self.df_val = self.data.get_data()
        self.model_dir = Path(model_dir)

    def run(self):
        model_names = ['best_model_kNN', 'best_model_RF', 'best_model_XGBoost']

        for model_name in model_names:
            model_dirs = list(self.model_dir.glob(f"**/{model_name}/"))
            for model_dir in model_dirs:
                print(f"Running inference for model: {model_dir.name}")
                self._evaluate_model(model_dir)

    #Adapt for multitask prediction
    def _evaluate_model(self, model_dir):

        num_turbines = self.config.data_settings.num_turbines
        y_pred_org = np.zeros((len(self.df_val), num_turbines))
        y_val_org = np.zeros_like(y_pred_org)

        if self.config.data_settings.multitask:
            model = joblib.load(model_dir / 'multitask_model.pkl')

            x_val = self.df_val[self.data.x_columns].to_numpy()
            y_val = self.df_val[self.data.y_columns].to_numpy()

            col_indices = [self.data.scale_cols.index(col) for col in self.data.y_columns]
 
            y_pred = model.predict(x_val)

            y_pred_org = y_pred * (self.data.scaler.data_max_[col_indices] - self.data.scaler.data_min_[col_indices]) + self.data.scaler.data_min_[col_indices]
            y_val_org = y_val * (self.data.scaler.data_max_[col_indices] - self.data.scaler.data_min_[col_indices]) + self.data.scaler.data_min_[col_indices]

        else:
            for i in range(num_turbines):
                model = joblib.load(model_dir / f'model_turbine_{i}.pkl')
                self.data.get_column_names_training(i)

                x_val = self.df_val[self.data.x_columns].to_numpy()
                y_val = self.df_val[self.data.y_columns].to_numpy()

                col_indices = [self.data.scale_cols.index(col) for col in self.data.y_columns]
                y_pred = model.predict(x_val)
                y_pred = model.predict(x_val)

                y_pred_org[:, i] = y_pred.ravel() * (self.data.scaler.data_max_[col_indices] - self.data.scaler.data_min_[col_indices]) + self.data.scaler.data_min_[col_indices]
                y_val_org[:, i] = y_val.ravel() * (self.data.scaler.data_max_[col_indices] - self.data.scaler.data_min_[col_indices]) + self.data.scaler.data_min_[col_indices]

        result_dict = self.get_error(y_val_org, y_pred_org)
        print(f"RMSE per turbine: {result_dict['rmse_per_turbine']}")

        # --- Plotting ---
        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        axes = axes.flatten()
        for i in range(num_turbines):
            ax = axes[i]
            ax.scatter(y_pred_org[:, i], y_val_org[:, i], s=0.5)
            ax.set_title(f'Turbine {i:02d} | RMSE: {result_dict["rmse_per_turbine"][i]:.2f}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')

        plt.tight_layout()
        fig.suptitle(f'Avg RMSE per turbine: {np.mean(result_dict["rmse_per_turbine"]):.2f}', fontsize=16, y=1.02)

        # --- Saving results ---
        if config.data_settings.hourly_data:
            save_path = model_dir / "inference_plot_hourly.png"
            file_path_save = model_dir / "results_hourly.json"
        else:
            save_path = model_dir / "inference_plot.png"
            file_path_save = model_dir / "results.json"

        print(save_path)
        fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved inference plot to {save_path}")

        with open(file_path_save, "w") as f:
            json.dump(result_dict, f, indent=4)
        print(f"Saved .json file with results {file_path_save}")

    def get_error(self,y_val_org, y_pred_org):
        mask = y_val_org != 0
        error = (y_pred_org - y_val_org) * mask

        mse_per_row = (error ** 2).mean(axis=1)
        mae_per_row = np.abs(error).mean(axis=1)
        rmse_per_turbine = np.sqrt(np.mean(error**2, axis=0))

        total_pred = (y_pred_org * mask).sum(axis=1)
        total_true = (y_val_org * mask).sum(axis=1)

        farm_mape = np.mean(np.abs((total_true - total_pred) / total_true)) * 100

        return {
            "mse_per_row": mse_per_row.tolist(), 
            "mae_per_row": mae_per_row.tolist(),
            "rmse_per_turbine": rmse_per_turbine.tolist(),
            "mape_per_row": farm_mape.tolist()
        }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run inference on saved models")
    parser.add_argument('--model_dir', '-m', type=str,
                        help="Path to directory containing models .pkl files", default='ml_framework/config_files')
    parser.add_argument('--hourly_data_override', type=int, choices=[0, 1], default=None,
                    help="Override hourly_data setting in config (true/false)")

    args = parser.parse_args()

    config_dir = Path(args.model_dir)
    for subdir in config_dir.iterdir():
        print('Running directory', subdir)
        if subdir.is_dir():
            print(subdir)
            yaml_files = [f for f in subdir.glob("config.yml") if not f.name.startswith('.')]
            if yaml_files:
                config = Box.from_yaml(filename=yaml_files[0], Loader=yaml.FullLoader)

                if args.hourly_data_override is not None:
                    config.data_settings.hourly_data = bool(args.hourly_data_override)
                    print(f"Overriding hourly_data with: {config.data_settings.hourly_data}")

                inference_runner = Inference(config, subdir)
                inference_runner.run()
