"""
Live Trading Configuration
===========================
All coin configs, trading params, API keys
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / '.env')

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
        'long_model': 'btc_cnn_model.pt',
        'short_model': 'btc_short_cnn_model.pt',
        'long_scaler': 'btc_feature_scaler.joblib',
        'short_scaler': 'btc_short_feature_scaler.joblib',
        'long_features': 'btc_features.json',
        'short_features': 'btc_short_features.json',
        'timeframes': ['4h', '1d', '1w'],
        'long_conf': 0.60,
        'short_conf': 0.58,   # V6: wide SL, 85.7% combined WR
        'data_start': '2017-01-01',
    },
    'ETH': {
        'pair': 'ETH/USDT',
        'long_model': 'eth_cnn_model.pt',
        'short_model': 'eth_short_cnn_model.pt',
        'long_scaler': 'eth_feature_scaler.joblib',
        'short_scaler': 'eth_short_feature_scaler.joblib',
        'long_features': 'eth_features.json',
        'short_features': 'eth_short_features.json',
        'timeframes': ['4h', '1d', '1w'],
        'long_conf': 0.60,
        'short_conf': 0.60,   # ETH SHORT: 58.1% WR @ 50%
        'data_start': '2018-01-01',
    },
    'SOL': {
        'pair': 'SOL/USDT',
        'long_model': 'sol_cnn_model.pt',
        'short_model': 'sol_short_cnn_model.pt',
        'long_scaler': 'sol_feature_scaler.joblib',
        'short_scaler': 'sol_short_feature_scaler.joblib',
        'long_features': 'sol_features.json',
        'short_features': 'sol_short_features.json',
        'timeframes': ['4h', '1d', '1w'],
        'long_conf': 0.65,
        'short_conf': 0.55,   # SOL SHORT: 85.7% WR @ 55%
        'data_start': '2020-08-01',
    },
    'DOGE': {
        'pair': 'DOGE/USDT',
        'long_model': 'doge_cnn_model.pt',
        'short_model': 'doge_short_cnn_model.pt',
        'long_scaler': 'doge_feature_scaler.joblib',
        'short_scaler': 'doge_short_feature_scaler.joblib',
        'long_features': 'doge_features.json',
        'short_features': 'doge_short_features.json',
        'timeframes': ['4h', '1d', '1w'],
        'long_conf': 0.59,
        'short_conf': 0.55,   # DOGE SHORT: 60% WR
        'data_start': '2019-07-01',
    },
    'AVAX': {
        'pair': 'AVAX/USDT',
        'long_model': 'avax_cnn_model.pt',
        'short_model': 'avax_short_cnn_model.pt',
        'long_scaler': 'avax_feature_scaler.joblib',
        'short_scaler': 'avax_short_feature_scaler.joblib',
        'long_features': 'avax_features.json',
        'short_features': 'avax_short_features.json',
        'timeframes': ['4h', '1d', '1w'],
        'long_conf': 0.55,
        'short_conf': 0.55,   # AVAX SHORT: 70% WR, +50.9%!
        'data_start': '2020-09-01',
    },
}

# Trading parameters
TRADING = {
    'tp_pct': 0.015,
    'sl_pct': 0.0075,
    'position_size_pct': 0.18,     # 18% per coin (5 coins = 90% max exposure)
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
