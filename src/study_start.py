import os

import optuna


storage_name = "sqlite:///mydb_lstm_crypto_1min_w_lag.db"


study = optuna.create_study(
    directions=["minimize"],
    study_name="lstm_crypto_study_only_kernel",
    storage=storage_name,
    load_if_exists=True,
)
