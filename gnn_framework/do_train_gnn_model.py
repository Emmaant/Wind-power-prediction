from torch_geometric.loader import DataLoader
from box import Box
from datetime import datetime
import os, torch, optuna, argparse, yaml, wandb
from tqdm import tqdm
from processor_settings.optimizer import get_optimizer
from gnn_framework.gnn_architecture import *


def build_model(config):
    model = WindFarmGNN(config)
    if config.io_settings.pretrained_model:
        checkpoint = torch.load(config.io_settings.pretrained_model, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        for param in model.encoder.parameters():
            param.requires_grad = False
        for param in model.processor.parameters():
            param.requires_grad = False
        model.decoder = Decoder(
            config.encoder_settings.node_latent_dim,
            config.decoder_settings.node_dec_mlp_layers,
            config.decoder_settings.output_dim
        )
    return model

def train_one_epoch(model, optimizer, train_loader, device):
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        batch_loss = model.compute_loss(data.y, out)
        mask = data.y[:, 0] != 0
        masked_y = data.y[mask, 0]
        masked_y_pred = out[mask, 0]
        total_loss += torch.nn.functional.mse_loss(
            masked_y, masked_y_pred, reduction='mean').item()
        batch_loss.backward()
        optimizer.step()
    return total_loss / len(train_loader)

def evaluate(model, validate_loader, device):
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for data in validate_loader:
            data = data.to(device)
            out = model(data)
            mask = data.y[:, 0] != 0
            masked_y = data.y[mask, 0]
            masked_y_pred = out[mask, 0]
            if masked_y.numel() == 0:
                total_val_loss += model.compute_loss(data.y, out).item()
            else:
                total_val_loss += torch.nn.functional.mse_loss(
                    masked_y, masked_y_pred, reduction='mean'
                ).item()
    return total_val_loss / len(validate_loader)


def train(config, graphs_train, validate_loader):
    patience = config.hyperparameters.early_stopping_patience
    train_loader = DataLoader(
        graphs_train,
        batch_size=config.hyperparameters.batch_size,
        shuffle=True,
        exclude_keys=[],
        num_workers=config.run_settings.num_t_workers,
        pin_memory=True,
        persistent_workers=config.run_settings.num_t_workers != 0
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_name = config.io_settings.directory.split('graphs/')[1]
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{run_name}_{current_time}"
    current_run_dir = os.path.join(config.io_settings.run_dir, run_name)
    os.makedirs(os.path.join(current_run_dir, 'trained_models'), exist_ok=True)
    wandb.init(project=config.io_settings.wandb_project_name, config=config, name=run_name, reinit=True)
    config.to_yaml(filename=os.path.join(current_run_dir, 'config.yml'))
    model = build_model(config).to(device)
    optimizer = get_optimizer(
        config.model_settings.processor_settings.optimizer,
        model.parameters(),
        float(config.hyperparameters.start_lr)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.hyperparameters.epochs
    )

    print(f'Starting run {run_name} on {next(model.parameters()).device}')
    pbar = tqdm(total=config.hyperparameters.epochs)
    pbar.set_description('Training')

    for epoch in range(config.hyperparameters.epochs):
        train_loss = train_one_epoch(model, optimizer, train_loader, device)
        wandb.log({"epoch": epoch, "train loss": train_loss})

        if epoch == 0:
            torch.save({'model_state_dict': model.state_dict()}, os.path.join(current_run_dir, 'trained_models', 'best.pt'))

        if (epoch + 1) % config.io_settings.save_epochs == 0:
            torch.save({'model_state_dict': model.state_dict()}, os.path.join(current_run_dir, 'trained_models', f'e{epoch + 1}.pt'))

        validation_loss = evaluate(model, validate_loader, device)
        wandb.log({"validation loss": validation_loss}, step=epoch + 1)

        if scheduler.get_last_lr()[0] > float(config.hyperparameters.min_lr):
            scheduler.step()

        if epoch == 0:
            best_validation_loss = validation_loss
        elif validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save({'model_state_dict': model.state_dict()}, os.path.join(current_run_dir, 'trained_models', 'best.pt'))
            patience = config.hyperparameters.early_stopping_patience
        else:
            patience -= 1
            if patience == 0:
                break
        pbar.set_postfix({'Train Loss': f'{train_loss:.8f}', 'Validation Loss': f'{validation_loss:.8f}'})
        pbar.update(1)

    wandb.run.summary["final loss"] = validation_loss
    config.best_validation_loss = best_validation_loss
    config.to_yaml(filename=os.path.join(current_run_dir, 'config.yml'))
    return best_validation_loss

def optuna_objective(trial, config, graphs_train, validate_loader):
    config.hyperparameters.start_lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    config.hyperparameters.batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    config.model_settings.num_layers = trial.suggest_int("num_layers", 2, 10)
    config.encoder_settings.node_enc_mlp_layers = [
        trial.suggest_categorical("node_enc_1", [64,128,256,512]),
        trial.suggest_categorical("node_enc_2", [64,128,256]),
        trial.suggest_categorical("node_enc_3", [64,128,256])
    ]
    config.encoder_settings.edge_enc_mlp_layers = [
        trial.suggest_categorical("edge_enc_1", [64,128,256,512]),
        trial.suggest_categorical("edge_enc_2", [64,128,256]),
        trial.suggest_categorical("edge_enc_3", [64,128,256])
    ]
    config.decoder_settings.node_dec_mlp_layers = [
        trial.suggest_categorical("decoder_1", [32, 64, 128]),
        trial.suggest_categorical("decoder_2", [32, 64, 128])
    ]
    latent_dim = trial.suggest_categorical("latent_dim", [64, 128, 256])
    config.decoder_settings.node_latent_dim = latent_dim
    config.decoder_settings.edge_latent_dim = latent_dim
    return train(config, graphs_train, validate_loader)

def train_models(config_path: str):
    config = Box.from_yaml(filename=config_path, Loader=yaml.FullLoader)
    for item in os.listdir(config.io_settings.path):
        config.io_settings.directory = os.path.join(config.io_settings.path, item)
        train_path = os.path.join(config.io_settings.directory, config.io_settings.train_graphs_path)
        val_path = os.path.join(config.io_settings.directory, config.io_settings.validate_graphs_path)
        graphs_train = torch.load( train_path, weights_only=False)
        graphs_val = torch.load(val_path, weights_only = False)
        config.edge_attr_dim = graphs_train[0].edge_attr.shape[1]
        config.feature_dim = graphs_train[0].x.shape[1]
        config.decoder_settings.output_dim = graphs_train[0].y.shape[1]
        validate_loader = DataLoader(
            graphs_val,
            batch_size=1,
            shuffle=False,
            exclude_keys=[],
            num_workers=config.run_settings.num_v_workers,
            pin_memory=True,
            persistent_workers=config.run_settings.num_v_workers != 0
        )
        wandb.login()
        if config.hyper_tuning_settings.tune:
            study = optuna.create_study(direction="minimize")
            study.optimize(lambda trial: optuna_objective(trial, config, graphs_train, validate_loader), n_trials=config.hyper_tuning_settings.n_trials)
            print("Best trial:", study.best_trial.params)
        else:
            train(config, graphs_train, validate_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_config', '-c', help="path to the yaml config file", type=str,required=False, default='gnn_framework/config.yml')
    train_models(config_path=parser.parse_args().yaml_config)