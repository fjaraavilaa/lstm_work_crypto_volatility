# LSTM Cryptocurrency Volatility Prediction

This is a Deep Kernel Learning project I decided to do after my PhD. I was just annoyed at the fact that I never found GPs used with LSTMs. Might sound a bit stubborn but I thought it could help someone (like me).

##  Project Overview

This project implements an end-to-end pipeline for cryptocurrency Returns prediction using Long Short-Term Memory (LSTM) neural networks. It includes data collection, feature engineering, model training, and prediction capabilities.

I am not very interested in the crypto usage. Please use things are you please.

##  Features

- **Multiple Data Sources**: Binance API (preferred) and yfinance (fallback)
- **Enhanced Data Quality**: Real-time, high-frequency data from the largest crypto exchange
- **Automated Data Collection**: Fetch cryptocurrency data with superior accuracy
- **Feature Engineering**: Calculate technical indicators and volatility measures
- **LSTM Model**: Deep learning architecture optimized for time series prediction
- **GPyTorch Integration**: Multi-Task Sparse Variational Gaussian Processes with uncertainty quantification
- **Hybrid Models**: Combine LSTM feature extraction with GP prediction
- **Comprehensive Pipeline**: From raw data to trained model
- **Model Evaluation**: Performance metrics and visualization tools


### ** Enhanced Volatility Features with Binance**
- **Parkinson Volatility**: Uses High-Low range for better estimation
- **Garman-Klass Volatility**: Incorporates OHLC data for accuracy  
- **Rogers-Satchell Volatility**: Drift-independent volatility estimator
- **Volume-Weighted Features**: True trading volume analysis
- **Order Book Insights**: Taker buy ratios and trade sizes

All these came from Claude. I am just doing a proof of concept. I do not care too much about this.

## 📁 Project Structure

It's still a bit messy. Consider it a to do.

##  Setup and Installation

### 1. Environment Setup

Make sure you have Python 3.10 installed and create a virtual environment:

```bash
# Create virtual environment
python3.10 -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate     # On Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
python -c "import torch; import pandas; import yfinance; print('✅ All dependencies installed successfully!')"
```

##  Usage

### Hyper-parameter search

If you want to run the hyper-parameter search specifically do the following.

```
python -m src.study_start
python -m src.optuna_trial_function  --symbols BTC-USD --input_study_features Returns Volume Count --device cuda 
```

If you want to do it with task farming

```
python -m src.study_start

srun --ntasks=1 --gres=gpu:1 --exact python -m src.optuna_trial_function  --symbols BTC-USD --input_study_features Returns Volume Count --device cuda &
srun --ntasks=1 --gres=gpu:1 --exact python -m src.optuna_trial_function  --symbols BTC-USD --input_study_features Returns Volume Count --device cuda &
wait
```





#### 3. Advanced Training Options

```bash
# Train with custom parameters
python main.py --mode train \
    --collect-data \
    --symbols BTC-USD ETH-USD ADA-USD DOT-USD LINK-USD \
    --sequence-length 45 \
    --epochs 150 \
    --batch-size 64 \
    --learning-rate 0.0005
```

#### 4. GPyTorch Enhanced Training

```bash
# Train hybrid LSTM-GP model with uncertainty quantification
python main.py --mode train-gp --model-type hybrid \
    --collect-data \
    --symbols BTC-USD ETH-USD ADA-USD DOT-USD \
    --epochs 100 \
    --gp-epochs 50

# Train pure Multi-Task Gaussian Process
python main.py --mode train-gp --model-type gp \
    --symbols BTC-USD ETH-USD ADA-USD \
    --num-inducing 150 \
    --gp-learning-rate 0.01

# Direct GPyTorch script usage
cd src
python train_with_gp.py --model-type hybrid --collect-data --symbols BTC-USD ETH-USD
```

#### 5. Enhanced Data Collection with Binance API

```bash
# High-frequency data collection (1-minute intervals)
python main.py --mode collect --data-source binance --interval 1m --symbols BTC-USD ETH-USD --period 7d

# Hourly data for more detailed volatility analysis
python main.py --mode collect --data-source binance --interval 1h --symbols BTC-USD ETH-USD SOL-USD --period 30d

# Compare data sources
python src/unified_data_collector.py

# Use Binance API with authentication for higher rate limits
python main.py --mode collect --data-source binance --binance-api-key YOUR_API_KEY --symbols BTC-USD ETH-USD

# Direct Binance data collector usage
cd src
python binance_data_collector.py
```

### Direct Script Usage

You can also run the individual scripts directly:

```bash
# Data collection
cd src
python data_collector.py

# Full training pipeline
cd src
python train_model.py --collect-data --symbols BTC-USD ETH-USD ADA-USD
```

## 🎛️ Configuration Options

### Data Collection Parameters

- `--data-source`: Data source to use ('binance', 'yfinance', 'auto') (default: 'auto')
- `--symbols`: Cryptocurrency symbols (e.g., BTC-USD, ETH-USD)
- `--period`: Data collection period for yfinance (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
- `--interval`: Data interval for Binance API ('1m', '5m', '15m', '1h', '4h', '1d')
- `--binance-api-key`: Binance API key for higher rate limits (optional)

### Model Parameters

- `--sequence-length`: Length of input sequences for LSTM (default: 30)
- `--hidden-size`: Number of hidden units in LSTM layers (default: 128)
- `--num-layers`: Number of LSTM layers (default: 2)
- `--dropout`: Dropout probability (default: 0.2)

### Training Parameters

- `--epochs`: Maximum number of training epochs (default: 100)
- `--batch-size`: Batch size for training (default: 32)
- `--learning-rate`: Learning rate for optimizer (default: 0.001)
- `--patience`: Patience for early stopping (default: 10)

### GPyTorch Parameters

- `--model-type`: Type of model ('lstm', 'gp', 'hybrid') (default: 'hybrid')
- `--num-inducing`: Number of inducing points for GP (default: 100)
- `--gp-epochs`: Number of epochs for pure GP training (default: 100)
- `--gp-learning-rate`: Learning rate for GP parameters (default: 0.01)
- `--gp-feature-dim`: Feature dimension for GP in hybrid model (default: 32)
- `--lstm-hidden-size`: LSTM hidden size in hybrid model (default: 64)
- `--gp-init-samples`: Samples for GP initialization (default: 1000)

## 📈 Model Architecture

### LSTM Model
The LSTM model includes:
- **Multi-layer LSTM**: 2-layer LSTM with configurable hidden size
- **Dropout Regularization**: Prevents overfitting
- **Fully Connected Layers**: Final prediction layers
- **Early Stopping**: Prevents overtraining

### GPyTorch Models

#### Multi-Task Sparse Variational Gaussian Process (SVGP)
- **Multi-Task Learning**: Simultaneously predict volatility for multiple cryptocurrencies
- **Uncertainty Quantification**: Provides prediction confidence intervals
- **Sparse Approximation**: Efficient training with inducing points
- **RBF Kernel**: Captures smooth temporal patterns
- **Multi-Task Kernel**: Models correlations between cryptocurrencies

#### Hybrid LSTM-GP Model
- **LSTM Feature Extractor**: Learns temporal representations from sequences
- **GP Predictor**: Makes final predictions with uncertainty from LSTM features
- **Joint Training**: End-to-end optimization of both components
- **Best of Both**: Combines LSTM's sequence modeling with GP's uncertainty

### Features Used

- **Price Data**: Open, High, Low, Close, Volume
- **Technical Indicators**: RSI, Moving Averages, Bollinger Bands, ATR
- **Volatility Measures**: Rolling standard deviations over multiple windows
- **Lagged Features**: Previous values for temporal patterns
- **Ratios and Normalized Values**: Price ratios and volume ratios

## 📊 Output and Results

After training, the model generates:

- **Trained Model**: Saved PyTorch model (.pth file)
- **Training History**: Loss curves and performance metrics
- **Metadata**: Model configuration and performance statistics
- **Visualizations**: Training history plots

Results are saved in the `models/` directory with timestamps.

## 🔬 Supported Cryptocurrencies

The system supports any cryptocurrency symbol available on Yahoo Finance:

- **Major Cryptocurrencies**: BTC-USD, ETH-USD, ADA-USD, DOT-USD
- **DeFi Tokens**: LINK-USD, UNI-USD, SUSHI-USD
- **Layer 1s**: SOL-USD, AVAX-USD, ATOM-USD
- **And many more**: Any symbol with `-USD` suffix

## 🧪 Model Performance

The model predicts cryptocurrency volatility with:

- **Multi-step ahead prediction capability**
- **Cross-cryptocurrency learning**
- **Technical indicator integration**
- **Robust evaluation metrics**

Performance metrics include MSE loss and visualization of predictions vs actual values.

## 🛠️ Development

### Adding New Features

1. **New Technical Indicators**: Add to `data_collector.py`
2. **Model Architectures**: Modify `lstm_model.py`
3. **Data Sources**: Extend `data_collector.py`

### Running Tests

If you're thinking of the tests, just go to the tests/ folder. I think I added a test for each function. If not please contact me or do a Pull Request.


## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is for educational and research purposes. Please ensure compliance with data usage policies and financial regulations in your jurisdiction.


---

**⚠️ Disclaimer**: This project is for educational purposes only. Cryptocurrency markets are highly volatile and unpredictable. Do not use these predictions for actual trading without thorough backtesting and risk management.

I ALSO DO NOT BELIEVE IN CRYPTOCURRENCIES BUT THEIR DATASET IS QUITE EXTENSIVE
