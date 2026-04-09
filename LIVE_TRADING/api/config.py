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
        'long_conf': 0.55,    # V3: optimizer Q1 2026 (59.1% WR, +83.2%)
        'short_conf': 0.50,   # V3: optimizer Q1 2026
        'long_meta_conf': 0.45,   # V3: XGBoost meta threshold LONG
        'short_meta_conf': 0.50,  # V3: XGBoost meta threshold SHORT
        'v3': True,
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
        'long_conf': 0.55,    # V3: optimizer Q1 2026 (62.5% WR, +46.2%)
        'short_conf': 0.50,   # V3: optimizer Q1 2026
        'long_meta_conf': 0.0,    # V3: meta LONG pass-through
        'short_meta_conf': 0.45,  # V3: XGBoost meta threshold SHORT
        'v3': True,
        'btc_influence': True,
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
        'long_conf': 0.55,    # V3: optimizer Q1 2026 (75.4% WR, +211.0%)
        'short_conf': 0.50,   # V3: optimizer Q1 2026
        'long_meta_conf': 0.0,    # V3: meta pass-through
        'short_meta_conf': 0.0,   # V3: meta pass-through
        'v3': True,
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
        'long_conf': 0.60,
        'short_conf': 0.55,   # Backtest original optimized
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
        'long_conf': 0.60,    # V3: optimizer Q1 2026 (60.5% WR, +64.8%)
        'short_conf': 0.50,   # V3: optimizer Q1 2026
        'v3': True,
        'data_start': '2020-09-01',
    },
    'XRP': {
        'pair': 'XRP/USDT',
        'long_model': 'xrp_cnn_model.pt',
        'short_model': 'xrp_short_cnn_model.pt',
        'long_scaler': 'xrp_feature_scaler.joblib',
        'short_scaler': 'xrp_short_feature_scaler.joblib',
        'long_features': 'xrp_features.json',
        'short_features': 'xrp_short_features.json',
        'timeframes': ['4h', '1d', '1w'],
        'long_conf': 0.75,    # V3: optimizer Q1 2026 (71.7% WR, +143.5%)
        'short_conf': 0.50,   # V3: optimizer Q1 2026
        'long_meta_conf': 0.55,   # V3: XGBoost meta threshold LONG
        'short_meta_conf': 0.0,   # V3: NoMeta SHORT
        'v3': True,
        'data_start': '2018-01-01',
    },
    'LINK': {
        'pair': 'LINK/USDT',
        'long_model': 'link_cnn_model.pt',
        'short_model': 'link_short_cnn_model.pt',
        'long_scaler': 'link_feature_scaler.joblib',
        'short_scaler': 'link_short_feature_scaler.joblib',
        'long_features': 'link_features.json',
        'short_features': 'link_short_features.json',
        'timeframes': ['4h', '1d', '1w'],
        'long_conf': 0.55,    # V3: optimizer Q1 2026 (60.7% WR, +18.8%)
        'short_conf': 0.55,   # V3: optimizer Q1 2026
        'long_meta_conf': 0.52,   # V3: XGBoost meta threshold LONG
        'short_meta_conf': 0.50,  # V3: XGBoost meta threshold SHORT
        'v3': True,
        'data_start': '2017-12-01',
    },
    'ADA': {
        'pair': 'ADA/USDT',
        'long_model': 'ada_cnn_model.pt',
        'short_model': 'ada_short_cnn_model.pt',
        'long_scaler': 'ada_feature_scaler.joblib',
        'short_scaler': 'ada_short_feature_scaler.joblib',
        'long_features': 'ada_features.json',
        'short_features': 'ada_short_features.json',
        'timeframes': ['4h', '1d', '1w'],
        'long_conf': 0.65,
        'short_conf': 0.55,   # Backtest original optimized
        'bear_sma50': -0.12,  # Per-coin override (backtest original)
        'bear_sma20': -0.05,  # Per-coin override (backtest original)
        'data_start': '2018-04-01',
    },
    'NEAR': {
        'pair': 'NEAR/USDT',
        'long_model': 'near_cnn_model.pt',
        'short_model': 'near_short_cnn_model.pt',
        'long_scaler': 'near_feature_scaler.joblib',
        'short_scaler': 'near_short_feature_scaler.joblib',
        'long_features': 'near_features.json',
        'short_features': 'near_short_features.json',
        'timeframes': ['4h', '1d', '1w'],
        'long_conf': 0.65,    # V3: optimizer Q1 2026 (64.9% WR, +43.4%)
        'short_conf': 0.50,   # V3: optimizer Q1 2026
        'long_meta_conf': 0.0,    # V3: meta pass-through
        'short_meta_conf': 0.0,   # V3: meta pass-through
        'v3': True,
        'data_start': '2020-10-01',
    },
    'DOT': {
        'pair': 'DOT/USDT',
        'long_model': 'dot_cnn_model.pt',
        'short_model': 'dot_short_cnn_model.pt',
        'long_scaler': 'dot_feature_scaler.joblib',
        'short_scaler': 'dot_short_feature_scaler.joblib',
        'long_features': 'dot_features.json',
        'short_features': 'dot_short_features.json',
        'timeframes': ['4h', '1d', '1w'],
        'long_conf': 0.55,    # CNN LONG confidence threshold
        'short_conf': 0.55,   # CNN SHORT confidence threshold
        'long_meta_conf': 0.0,    # Meta pass-through
        'short_meta_conf': 0.0,   # Meta pass-through
        'v3': True,
        'data_start': '2020-08-20',
    },
    'FIL': {
        'pair': 'FIL/USDT',
        'long_model': 'fil_cnn_model.pt',
        'short_model': 'fil_short_cnn_model.pt',
        'long_scaler': 'fil_feature_scaler.joblib',
        'short_scaler': 'fil_short_feature_scaler.joblib',
        'long_features': 'fil_features.json',
        'short_features': 'fil_short_features.json',
        'timeframes': ['4h', '1d', '1w'],
        'long_conf': 0.60,    # Backtest Q1 2026: 60% WR, +5.64%
        'short_conf': 0.55,   # CNN SHORT confidence threshold
        'long_meta_conf': 0.0,    # Meta pass-through
        'short_meta_conf': 0.0,   # Meta pass-through
        'v3': True,
        'data_start': '2020-10-15',
    },
}

# Trading parameters
TRADING = {
    'tp_pct': 0.015,      # Default for non-BTC coins
    'sl_pct': 0.0075,     # BTC uses symmetric ATR in signal_generator
    'position_size_pct': 0.10,     # 10% per coin (9 coins = 90% max exposure)
    'trading_fee': 0.001,
    'slippage': 0.0005,
    'use_dynamic_tp_sl': True,
    'max_hold_days': 10,
    'leverage': 1,                  # No leverage for testnet
}

# Filter parameters
FILTERS = {
    'max_consecutive_losses': 2,    # V3: optimizer best for all coins
    'cooldown_days': 2,             # V3: optimizer best (was 5)
    'min_momentum_alignment': 1,    # At least 1/3 TFs bullish
    'max_volatility_regime': 2.5,
    'min_adx': 15,
    'bear_sma50_threshold': -0.05,  # -5% below SMA50 (default)
    'bear_sma20_threshold': -0.02,  # -2% below SMA20 (backtest original)
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

# API Settings (used by main.py)
class settings:
    HOST = '0.0.0.0'
    PORT = 8080
    DEBUG = False
    CORS_ORIGINS = ['*']
    API_KEY = os.getenv('API_KEY', '098e53ee1afd8cbb5079c7ed6321f7f3')
    JWT_SECRET = os.getenv('JWT_SECRET', 'crypto-adviser-secret-key-2026')
    JWT_ALGORITHM = 'HS256'
    JWT_EXPIRATION = 3600
