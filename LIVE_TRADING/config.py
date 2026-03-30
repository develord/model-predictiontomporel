"""
Live Trading Configuration
===========================
All coin configs, trading params, API keys
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
PROJECT_DIR = BASE_DIR.parent
MODEL_DIR = BASE_DIR / 'models'
STATE_DIR = BASE_DIR / 'state'
LOG_DIR = BASE_DIR / 'logs'

# Binance Futures Testnet API
BINANCE_TESTNET_KEY = os.getenv('BINANCE_TESTNET_KEY', '')
BINANCE_TESTNET_SECRET = os.getenv('BINANCE_TESTNET_SECRET', '')

# Coin configurations
COINS = {
    'BTC': {
        'pair': 'BTC/USDT',
        'model_file': 'btc_cnn_model.pt',
        'scaler_file': 'btc_feature_scaler.joblib',
        'features_file': 'btc_features.json',
        'pipeline': 'simple',       # simple = BTC-style 02_feature_engineering
        'timeframes': ['4h', '1d', '1w'],
        'confidence_threshold': 0.60,
        'data_start': '2017-01-01',
    },
    'ETH': {
        'pair': 'ETH/USDT',
        'model_file': 'eth_cnn_model.pt',
        'scaler_file': 'eth_feature_scaler.joblib',
        'features_file': 'eth_features.json',
        'pipeline': 'simple',       # Retrain with simple pipeline for consistency
        'timeframes': ['4h', '1d', '1w'],
        'confidence_threshold': 0.60,
        'data_start': '2018-01-01',
    },
    'SOL': {
        'pair': 'SOL/USDT',
        'model_file': 'sol_cnn_model.pt',
        'scaler_file': 'sol_feature_scaler.joblib',
        'features_file': 'sol_features.json',
        'pipeline': 'simple',
        'timeframes': ['4h', '1d', '1w'],
        'confidence_threshold': 0.65,
        'data_start': '2020-08-01',
    },
    'DOGE': {
        'pair': 'DOGE/USDT',
        'model_file': 'doge_cnn_model.pt',
        'scaler_file': 'doge_feature_scaler.joblib',
        'features_file': 'doge_features.json',
        'pipeline': 'simple',
        'timeframes': ['4h', '1d', '1w'],
        'confidence_threshold': 0.59,
        'data_start': '2019-07-01',
    },
    'XRP': {
        'pair': 'XRP/USDT',
        'model_file': 'xrp_cnn_model.pt',
        'scaler_file': 'xrp_feature_scaler.joblib',
        'features_file': 'xrp_features.json',
        'pipeline': 'simple',
        'timeframes': ['4h', '1d', '1w'],
        'confidence_threshold': 0.55,
        'data_start': '2018-01-01',
    },
    'AVAX': {
        'pair': 'AVAX/USDT',
        'model_file': 'avax_cnn_model.pt',
        'scaler_file': 'avax_feature_scaler.joblib',
        'features_file': 'avax_features.json',
        'pipeline': 'simple',
        'timeframes': ['4h', '1d', '1w'],
        'confidence_threshold': 0.55,
        'data_start': '2020-09-01',
    },
}

# Trading parameters
TRADING = {
    'tp_pct': 0.015,
    'sl_pct': 0.0075,
    'position_size_pct': 0.15,     # 15% per coin (6 coins = 90% max exposure)
    'trading_fee': 0.001,
    'slippage': 0.0005,
    'use_dynamic_tp_sl': True,
    'max_hold_days': 10,
    'leverage': 1,                  # No leverage for testnet
}

# Filter parameters
FILTERS = {
    'max_consecutive_losses': 2,
    'cooldown_days': 5,
    'min_momentum_alignment': 1,    # At least 1/3 TFs bullish
    'max_volatility_regime': 2.5,
    'min_adx': 15,
    'bear_sma50_threshold': -0.05,  # -5% below SMA50
    'bear_sma20_threshold': -0.03,  # -3% below SMA20
    'max_trend_score': -3,          # Block if trend_score below this
}

# Buffer sizes for rolling OHLCV data
BUFFER_SIZES = {
    '15m': 1000,  # ~10 days for TP/SL monitoring
    '4h': 500,    # ~80 days
    '1d': 300,    # ~300 days for indicator warmup
    '1w': 100,    # ~2 years
}

# Sequence length for CNN
SEQUENCE_LENGTH = 30

# WebSocket
WS_RECONNECT_DELAY = 5  # seconds
WS_MAX_RECONNECT_DELAY = 60

# Prediction schedule
PREDICTION_TIMEFRAME = '1d'  # Predict on daily close
MONITOR_TIMEFRAME = '15m'    # Monitor TP/SL on 15min candles
