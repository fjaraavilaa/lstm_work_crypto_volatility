import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from gpytorch.likelihoods import likelihood
from gpytorch.mlls import VariationalELBO
import json

from src.gpytorch_model import DeepKernelSVGP, LatentSVGP

logging.basicConfig(level=logging.INFO)

def compute_prediction_metrics(y_true: torch.Tensor, y_pred: torch.Tensor):
    """Compute MSE and MAE between predictions and targets.
 
    Args:
        y_true (torch.Tensor): Ground truth values.
        y_pred (torch.Tensor): Model predictions, same shape as y_true.
 
    Returns:
        tuple: (mse, mae)
    """
    y_true = y_true.detach().flatten().float()
    y_pred = y_pred.detach().flatten().float()
 
    mse = F.mse_loss(y_pred, y_true).item()
    mae = F.l1_loss(y_pred, y_true).item()
 
    return mse, mae

def train_lstm_model(model: nn.Module, train_data_loader: DataLoader, val_data_loader: DataLoader, epochs: int, lr: float):

    #train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    #val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_data_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = sum(criterion(model(inputs), targets) for inputs, targets in val_data_loader) / len(val_data_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {val_loss.item()}")
    return model

def train_with_gp(model: DeepKernelSVGP, train_data: TensorDataset, 
                  val_data: TensorDataset, batch_size: int, epochs: int,
                  likelihood: likelihood, optimizer: optim.Optimizer,
                  min_delta: float, patience: int, monitor: str = 'val',
                  save_suffix: str = '', save_metrics: bool = False, 
                  device: torch.device = torch.device('cpu')) -> LatentSVGP:
    """This funtions trains an SVGP model 

    Args:
        model (DeepKernelSVGP): The model to be trained with
        all the specifications already done.
        train_data (TensorDataset): Training Data already passed
        as a tensor
        val_data (TensorDataset): Validation Data already passed
        as a tensor
        batch_size (int): Batch size (can be optimized)
        epochs (int): Epochs to run
        likelihood (likelihood): Likelihood from gpflow used.
        Usually it'll be gaussian
        optimizer (optim.Optimizer): optimizer from pytorch
        min_delta (float): minimum permitted difference
        patience (int): How many trials to permit

    Returns:
        LatentSVGP: Fully trained model.
    """
    if monitor not in ("val", "train"):
        raise ValueError(f"monitor must be 'val' or 'train', got {monitor!r}")
    if monitor == "val" and val_data is None:
        raise ValueError("monitor='val' requires val_data to be provided")
    train_loader = DataLoader(train_data, batch_size, shuffle = False)
    val_loader = DataLoader(val_data, batch_size, shuffle = False) if val_data is not None else None
    mll = VariationalELBO(
        likelihood.to(device), model.inferential_process.to(device), num_data = len(train_data)
    )

    model.train()
    best_val = float('inf')
    
    likelihood.train()
    logging.info("Starting GP training...")
    train_losses = []
    val_losses = []
    mse_losses = []
    mae_losses = []
    mse_losses_val = []
    mae_losses_val = []
    for epoch in range(epochs):
        batch_losses = []
        for x_batch, y_batch in train_loader:
            #logging.info(f"Epoch {epoch + 1}/{epochs}, Batch Loss: {mll(model(x_batch), y_batch).item()}")
            #logging.info(f"Epoch {epoch + 1}/{epochs}, Input shape: {x_batch.shape}, Target shape: {y_batch.shape}")
            optimizer.zero_grad()
            outputs = model(x_batch.to(device))
            loss = -mll(outputs, y_batch.to(device)).to(device)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        train_loss = sum(batch_losses) / len(batch_losses)
        logging.info(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}")
        train_losses.append(train_loss)
        model.eval()
        likelihood.eval()
        if val_loader is not None:
            with torch.no_grad():
                val_loss = sum(-mll(model(x_batch.to(device)), y_batch.to(device)) for x_batch, y_batch in val_loader) / len(val_loader)
            val_losses.append(val_loss.item())
        
        
        mse_train_local, mae_train_local = compute_prediction_metrics(
            torch.cat([y for _, y in train_loader]), 
            torch.cat([model(x).mean for x, _ in train_loader])
        )
        mse_losses.append(mse_train_local)
        mae_losses.append(mae_train_local)
        if val_loader is not None:
            mse_val_local, mae_val_local = compute_prediction_metrics(
                torch.cat([y for _, y in val_loader]), 
                torch.cat([model(x).mean for x, _ in val_loader])
            )
            mse_losses_val.append(mse_val_local)
            mae_losses_val.append(mae_val_local)
            logging.info(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss.item():.4f}, MSE Train: {mse_train_local:.4f}, MAE Train: {mae_train_local:.4f}, MSE Val: {mse_val_local:.4f}, MAE Val: {mae_val_local:.4f}")

        current_metric = val_loss if monitor == "val" else train_loss
 
        if current_metric < best_val - min_delta:
            best_val = current_metric
            best_metrics = {
                "epoch": epoch,
                "best_elbo": current_metric.item() if torch.is_tensor(current_metric) else current_metric,
                "mse_train": mse_train_local,
                "mae_train": mae_train_local,
                "mse_val": mse_val_local if val_loader is not None else None,
                "mae_val": mae_val_local if val_loader is not None else None,
                }
            best_state = {
                "model": model.state_dict(),
                "likelihood": likelihood.state_dict()
            }
            counter = 0
        else:
            counter += 1
        
        with open(f"trained_model_{save_suffix}.pt", "wb") as f:
            torch.save(best_state, f)
        
        if counter >= patience:
            logging.info(f"Early Stopping at Epoch {epoch} (monitor={monitor})")
            break
        logging.info(f"Validation Loss: {val_loss.item()}")

    
    metrics = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "mse_losses_train": mse_losses,
        "mae_losses_train": mae_losses,
        "mse_losses_val": mse_losses_val,
        "mae_losses_val": mae_losses_val
    }
    with open(f"training_metrics_{save_suffix}.json", "w") as f:
        json.dump(metrics, f)

    return (model, metrics, best_metrics)

