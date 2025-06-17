"""
Script for running inference on validation data for MLP models
"""

import argparse
import json
from pathlib import Path
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
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
        self.model_dir = Path(model_dir)
        self.device = torch.device("cpu")  # force CPU

        self.data = Data(config)
        self.data.get_data_columns()
        self.df_tr, self.df_val = self.data.get_data()

        self.target_cols = self.data.target_columns
        self.input_cols = self.data.data_columns

        self.scaler = self.data.scaler  # assumes fitted MinMaxScaler
        self.col_indices = [self.data.scale_cols.index(col) for col in self.target_cols]

    def run(self):
        model_dirs = list(self.model_dir.glob("**/best_MLP/"))
        for mdl_dir in model_dirs:
            print(f"Running inference for: {mdl_dir.parent.name}")
            self._evaluate_model(mdl_dir)

    def _evaluate_model(self, model_dir):
        pt_files = list(model_dir.glob("*.pt"))
        pt_files = [f for f in pt_files if not f.name.startswith('.')]
        if len(pt_files) == 0:
            print(f"No .pt model file found in: {model_dir}")
            return
        elif len(pt_files) > 1:
            print(f"Multiple .pt files found in: {model_dir}, using the first one: {pt_files[0].name}")

        model_path = pt_files[0]
        print(f"Using model file: {model_path.name}")

        x_val = torch.tensor(self.df_val[self.input_cols].values, dtype=torch.float32)
        y_val = torch.tensor(self.df_val[self.target_cols].values, dtype=torch.float32)
        
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
            y_pred = model(x_val.to(self.device)).cpu().numpy()

        y_val_np = y_val.numpy()

        # Rescale
        y_pred_org = y_pred * (self.scaler.data_max_[self.col_indices] - self.scaler.data_min_[self.col_indices]) + self.scaler.data_min_[self.col_indices]
        y_val_org = y_val_np * (self.scaler.data_max_[self.col_indices] - self.scaler.data_min_[self.col_indices]) + self.scaler.data_min_[self.col_indices]

        result_dict = self.get_error(y_val_org, y_pred_org)

        print(f"RMSE per turbine: {result_dict['rmse_per_turbine']}")

        self.plot_results(y_pred_org, y_val_org, result_dict, model_dir)

        result_file = model_dir / ("results_hourly.json" if self.config.data_settings.hourly_data else "results.json")
        with open(result_file, "w") as f:
            json.dump(result_dict, f, indent=4)
        print(f"Saved results to {result_file}")

    def plot_results(self, y_pred_org, y_val_org, result_dict, model_dir):
        num_turbines = y_pred_org.shape[1]
        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        axes = axes.flatten()
        for i in range(num_turbines):
            ax = axes[i]
            ax.scatter(y_pred_org[:, i], y_val_org[:, i], s=0.5)
            ax.set_title(f'Turbine {i:02d} | RMSE: {result_dict["rmse_per_turbine"][i]:.2f}')
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")

        plt.tight_layout()
        fig.suptitle(f'Avg RMSE per turbine: {np.mean(result_dict["rmse_per_turbine"]):.2f}', fontsize=16, y=1.02)
        plot_path = model_dir / ("inference_plot_hourly.png" if self.config.data_settings.hourly_data else "inference_plot.png")
        fig.savefig(plot_path, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved plot to {plot_path}")

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
    parser = argparse.ArgumentParser(description="Run inference for PyTorch MLP model on CPU")
    parser.add_argument('--model_dir', '-m', type=str, required=True, help="Path to root directory with config and trained model")
    parser.add_argument('--hourly_data_override', type=int, choices=[0, 1], default=None, help="Override hourly_data setting in config")

    args = parser.parse_args()

    config_dir = Path(args.model_dir)
    for subdir in config_dir.iterdir():
        if subdir.is_dir():
            yaml_files = list(subdir.glob(".yml"))
            yaml_files = [f for f in subdir.glob("config_mlp.yml") if not f.name.startswith('.')]
            if yaml_files:
                print(yaml_files)
                config = Box.from_yaml(filename=yaml_files[0], Loader=yaml.FullLoader)

                if args.hourly_data_override is not None:
                    config.data_settings.hourly_data = bool(args.hourly_data_override)

                inference_runner = Inference(config, subdir)
                inference_runner.run()
