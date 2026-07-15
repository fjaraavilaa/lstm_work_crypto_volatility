import pandas as pd
from src.binance_data_collector import BinanceDataCollector
from src.data_preprocessor import DataPreprocessor

collector = BinanceDataCollector()

collector.get_available_symbols()
crypto_data = collector.fetch_crypto_data(symbols=['BTC-USD', 'ETH-USD'], interval='1m', days_back=365)

preprocessor = DataPreprocessor()
processed_data = preprocessor.prepare_features(crypto_data, drop_bad_values=False)
processed_data = preprocessor.scale_features(processed_data.set_index(['Date', 'Symbol']), scaler = 'minmax')
###CREATE SEQUENCES ALREADY GIVES A PROPER RESULT
###TRY REPLICATING THAT FUNCTION FOR THE LSTM BUILDER

sequences = preprocessor.create_sequences(processed_data.reset_index(), symbol = 'BTC-USD', 
                                          sequence_length=5, target_col='Returns',
                                          input_study_features = ['Returns', 'Volume'])




processed_data.loc[processed_data['Symbol'] == 'BTC-USD', 
                   ['Returns', 'Volume']].iloc[0:10]

stupid = preprocessor.prepare_lstm_data(
    processed_data.reset_index(), symbols=['BTC-USD'],
    sequence_length=5)
