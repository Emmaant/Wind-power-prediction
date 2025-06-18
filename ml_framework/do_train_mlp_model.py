"""
Script for training MLP model.

Configure training by providing directory of config file(s) in .yml format and specify number of trails to run when
hypertuning using optuna. Example file is availible in directory config_files_mlp
"""

import torch
import torch.nn as nn
import torch.optim as optim
import wandb, glob, os, yaml, argparse, optuna
from pathlib import Path
import shutil
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from box import Box

from utils.helpers_ml_models import *

def load_data(config):
    dataloader = Data(config)
    dataloader.get_data_columns()
    df_tr, df_val = dataloader.get_data()

    target_prefixes = config.data_settings.targets
    target_cols = [col for col in df_tr.columns if any(col.startswith(prefix) for prefix in target_prefixes)]

    x_tr = torch.tensor(df_tr.drop(columns=target_cols).values, dtype=torch.float32)
    y_tr = torch.tensor(df_tr[target_cols].values, dtype=torch.float32)

    x_val = torch.tensor(df_val.drop(columns=target_cols).values, dtype=torch.float32)
    y_val = torch.tensor(df_val[target_cols].values, dtype=torch.float32)

    config.input_dim = x_val.shape[1]
    config.output_dim = y_val.shape[1]

    train_dataset = TensorDataset(x_tr, y_tr)
    val_dataset = TensorDataset(x_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    return train_loader, val_loader

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
    
    def compute_loss(self,y,out):
        mask_first_column = (y[:, 0] != 0) 

        if mask_first_column.sum() == 0:
            return torch.tensor(1e-12, requires_grad=True)  
        y_masked = y[mask_first_column]
        out_masked = out[mask_first_column]
        return nn.functional.mse_loss(y_masked, out_masked, reduction='mean')

def train(config_dir: str):
    config_paths = sorted(glob.glob(os.path.join(config_dir, "*.yml")))

    for config_path in config_paths:
        config = Box.from_yaml(filename=config_path, Loader=yaml.FullLoader)
        train_loader,validate_loader = load_data(config)
        run_name = '_'.join([
                            str(config_path),    
                            str(config.input_dim), 'input_dim',
                            str(config.hidden_dim), 'hidden_dim',
                            str(config.output_dim), 'output_dim',
                            str(config.num_layers), 'num_layers',
                            str(config.lr), 'lr',
                            str(config.batch_size), 'batch_size'])

        current_run_dir = os.path.join(config.run_dir, run_name)
        os.makedirs(os.path.join(current_run_dir, 'trained_models'), exist_ok=True)

        wandb.init(
            project = 'ML_MLP',
            name = run_name,
            reinit = True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        patience = config.early_stopping_patience 
        
        model = MLP(config.input_dim, config.hidden_dim, config.output_dim, config.num_layers, config.dropout)
        model.to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=float(config.lr))

        num_t_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('Number of trainable parameters: {}'.format(num_t_params))

        print('Starting run {} on {}'.format(run_name, next(model.parameters()).device))
        pbar = tqdm(total=config.epochs)
        pbar.set_description('Training')

        for epoch in range(config.epochs):
            model.train()
            total_loss = 0

            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()
                out = model(x_batch)
                loss = model.compute_loss(y_batch, out)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            train_loss = total_loss / len(train_loader)
            wandb.log({"epoch": epoch,
                "train loss": train_loss})

            validation_loss = 0
            model.eval()
            with torch.no_grad():
                for x_batch, y_batch in validate_loader:
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)
                    out = model(x_batch)
                    validation_loss += model.compute_loss(y_batch, out).item()

            validation_loss = validation_loss / len(validate_loader)
            wandb.log({"validation loss": validation_loss}, step = epoch+1)


            if epoch == 0:
                best_validation_loss = validation_loss
            else:
                if validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    torch.save({'model_state_dict': model.state_dict()},
                                os.path.join(current_run_dir, 'trained_models', 'best.pt'))
                    
                    patience = config.early_stopping_patience 
                else:
                    patience -= 1
                    if patience == 0:
                        break

            pbar.set_postfix({'Train Loss': f'{train_loss:.8f}', 'Validation Loss': f'{validation_loss:.8f}'})
            pbar.update(1)
            
        config.best_validation_loss = best_validation_loss
        config.to_yaml(filename=os.path.join(current_run_dir, 'config.yml'))


def objective(trial, base_config: Box, config_path: str, current_run_dir: Path):
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128, 256])
    num_layers = trial.suggest_int("num_layers", 1, 11)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)

    config = base_config.copy()
    config.hidden_dim = hidden_dim
    config.num_layers = num_layers
    config.lr = lr
    config.batch_size = batch_size  # Ensure batch size is also used

    run_name = f"{os.path.basename(config_path).replace('.yml','')}_trial{trial.number}_hd{hidden_dim}_nl{num_layers}_bs{batch_size}_lr{lr:.1e}"
    wandb.init(
        project="ml_models_mlp",
        name=run_name,
        config=config.to_dict(),
        reinit=True,
        anonymous="allow"
    )

    train_loader, validate_loader = load_data(config)
    model = MLP(config.input_dim, hidden_dim, config.output_dim, num_layers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    patience = config.early_stopping_patience
    best_model_path = current_run_dir / f"trial_{trial.number}_best.pt"

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = model.compute_loss(yb, out)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_loss = total_loss / len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in validate_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                val_loss += model.compute_loss(yb, out).item()
        val_loss /= len(validate_loader)

        wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = config.early_stopping_patience
            torch.save({'model_state_dict': model.state_dict()}, best_model_path)
        else:
            patience -= 1
            if patience == 0:
                break
    
    trial.set_user_attr("model_path", str(best_model_path))
    wandb.finish()
    return best_val_loss

def remove_all_pt_files(directory):
    directory = Path(directory)
    for pt_file in directory.glob("*.pt"):
        pt_file.unlink()

def run_optuna(config_dir: str, n_trials: int = 30):
    config_dir = Path(config_dir)

    for subdir in config_dir.iterdir():
        print('Running directory', subdir)
        if subdir.is_dir():
            best_model_dir = subdir / "best_MLP"
            best_model_dir.mkdir(parents=True, exist_ok=True)

            yaml_files = list(subdir.glob("*.yml"))
            config = Box.from_yaml(filename=yaml_files[0], Loader=yaml.FullLoader)

            def objective_with_dir(trial):
                return objective(trial, config, yaml_files[0], subdir)

            study = optuna.create_study(direction="minimize")
            study_results = study.optimize(objective_with_dir, n_trials=n_trials)

            best_trial = study.best_trial
            model_path = best_trial.user_attrs["model_path"]

            shutil.move(str(model_path), best_model_dir)
            remove_all_pt_files(subdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_config', '-c', help="path to the yaml config file", type=str,required=False, default='ml_framework/config_files_mlp')
    parser.add_argument('--trails', '-n', help="specify how many trails to run using optuna", type=int,required=False, default=30)

    args = parser.parse_args()
    
    run_optuna(args.yaml_config, args.trails)
