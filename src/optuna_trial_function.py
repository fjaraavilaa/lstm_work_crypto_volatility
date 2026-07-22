"""
hyperparameter search for LSTM model using Optuna, with logging and error handling
author: @fjaraavila
"""

import json
from pathlib import Path

import optuna
import logging
import torch
from gpytorch.kernels import RBFKernel
from src.gpytorch_model import LatentSVGP, DeepKernelSVGP
from src.binance_data_collector import BinanceDataCollector
from src.data_preprocessor import DataPreprocessor
from src.deep_kernel import lstm_extractor_same_size_layers
from src.training_functions import train_with_gp
import argparse
from gpytorch.likelihoods import GaussianLikelihood

with open('validation/healthy_dates.json', 'r') as f:
    config_dates = json.load(f)

parser = argparse.ArgumentParser(
    description = "hard-coded arguments for the lstm hyper-parameter search"
)
parser.add_argument('--symbols', nargs='+', default=['BTC-USD'], help='List of symbols to fetch data for')
parser.add_argument('--input_study_features', nargs='+', default=['Returns'], help='List of features to use in the study')
parser.add_argument('--device' , type=str, default='cuda', help='Device to use for training (e.g., "cuda" or "cpu")')

DATES_START_STUDY = config_dates['train']['start']
DATES_END_STUDY = config_dates['test']['end']
SYMBOLS = parser.parse_args().symbols
INPUT_STUDY_FEATURES = parser.parse_args().input_study_features
DEVICE = parser.parse_args().device


def objective(trial):
    #Define hyperparameters to optimize
    lstm_shape_lag = trial.suggest_int('lstm_shape_lag', 10, 100)
    lstm_hidden_size = trial.suggest_int('lstm_hidden_size', 5, 20)
    batch_size = trial.suggest_int('batch_size', 16, 2500)
    num_layers = trial.suggest_int('num_layers', 1, 5)
    dropout_sug = trial.suggest_float('dropout', 0.0, 0.5)

    #start the data collector
    collector = BinanceDataCollector()
    interval = '1m'
    crypto_raw_data = collector.fetch_crypto_data(symbols = SYMBOLS, 
                                                  interval = interval, 
                                                  start_date = DATES_START_STUDY, 
                                                  end_date = DATES_END_STUDY)
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.prepare_features(crypto_raw_data, target_col='Returns', drop_bad_values = False)

    processed_data = preprocessor.scale_features(processed_data.set_index(['Date', 'Symbol'])[INPUT_STUDY_FEATURES].dropna(), 
                                                 scaler = 'minmax')
    # Build training and validation splits here from the API-fetched dataframe.
    lstm_data = preprocessor.prepare_lstm_data(processed_data.reset_index(), symbols = SYMBOLS,
                                               sequence_length = lstm_shape_lag,
                                               target_col = 'Returns',
                                               prediction_horizon=1,
                                               date_splits =  {
                                                   'train': config_dates['train'],
                                                   'validation': config_dates['validation'],
                                                   'test': config_dates['test']
                                                   },
                                               input_study_features = INPUT_STUDY_FEATURES
                                               )
    
    logging.info(f'LSTM matrix is of shape {lstm_data['X_train'].shape}')
    
    del crypto_raw_data
    del processed_data

    inducing_points_n = trial.suggest_int('inducing_points', 50, 1000)
    inducing_points = torch.randn(inducing_points_n, lstm_data['X_train'].shape[1],
                                  lstm_data['X_train'].shape[2])

    feature_extractor = lstm_extractor_same_size_layers(
        input_size = lstm_data['X_train'].shape[2],
        output_size = 1,
        hidden_size = lstm_hidden_size,
        num_layers = num_layers,
        dropout = dropout_sug
    )
    base_kernel = RBFKernel()
    inferential_model = LatentSVGP(
        inducing_points = feature_extractor(inducing_points.float()),
        base_kernel = base_kernel,
        device = torch.device(DEVICE),
        learn_inducing_locations=True
    )
    compiled_model = DeepKernelSVGP(
        feature_extractor = feature_extractor,
        inferential_model = inferential_model,
        )
    likelihood = GaussianLikelihood()

    optimizer = torch.optim.Adam([
        {'params': compiled_model.feature_extractor.parameters()},
        {'params': compiled_model.inferential_process.hyperparameters()},
        {'params': compiled_model.inferential_process.variational_parameters()},
        {'params': likelihood.parameters()},
    ], lr=0.01)
    
    train_data = torch.utils.data.TensorDataset(
        torch.from_numpy(lstm_data['X_train']).float(),
        torch.from_numpy(lstm_data['y_train']).float(),

    )
    val_data = torch.utils.data.TensorDataset(
        torch.from_numpy(lstm_data['X_val']).float(),
        torch.from_numpy(lstm_data['y_val']).float()
    )
    trained_model = train_with_gp(
        compiled_model,
        train_data,
        val_data,
        batch_size=batch_size,
        epochs=10000,
        likelihood=likelihood,
        optimizer=optimizer,
        min_delta=1e-5,
        patience=5,
        monitor='val',
        save_metrics=False
    )
    return trained_model[2]['mae_val'] # Return the validation loss for Optuna to minimize

storage_name = "sqlite:///mydb_lstm_crypto_1min_w_lag.db"

if __name__ == "__main__":
    study = optuna.load_study(
        study_name="lstm_crypto_study_only_kernel", storage=storage_name
    )
    study.optimize(objective, n_trials=100)
