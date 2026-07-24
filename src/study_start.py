import os

import optuna


storage_name = "sqlite:///mydb_lstm_crypto_1min_w_lag.db"


study = optuna.create_study(
    directions=["minimize"],
    study_name="lstm_crypto_study_only_kernel",
    storage=storage_name,
    load_if_exists=True,
)

study.enqueue_trial(
    {
        "lstm_shape_lag": 5,
        "lstm_hidden_size": 18,
        "batch_size": 2100,
        "num_layers": 2,
        "inducing_points": 647,
        "dropout": 0.3878821630867384
    }
)