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
        'long_conf': 0.60,    # Backtest v1: 44% LONG WR, +4.34%
        'short_conf': 0.55,   # Backtest v1: 53% SHORT WR
        'long_meta_conf': 0.0,    # Disabled — not used in validated backtest
        'short_meta_conf': 0.0,   # Disabled
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
        'long_conf': 0.60,    # Backtest v2: 46% LONG WR, 85% SHORT WR, +76.05%
        'short_conf': 0.55,   # Backtest v2
        'long_meta_conf': 0.0,    # Disabled — not used in validated backtest
        'short_meta_conf': 0.0,   # Disabled
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
        'long_conf': 0.60,    # Backtest v1 seed123: 54% LONG, 100% SHORT, +78.85%
        'short_conf': 0.55,   # Backtest v1 seed2024
        'long_meta_conf': 0.0,    # Disabled
        'short_meta_conf': 0.0,   # Disabled
        'v3': True,
        'data_start': '2020-08-01',
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
        'long_conf': 0.60,    # Backtest v2: 100% LONG (4/4), 60% SHORT, +42.53%
        'short_conf': 0.50,   # Backtest v2
        'long_meta_conf': 0.0,    # Disabled — not used in validated backtest
        'short_meta_conf': 0.0,   # Disabled
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
        'long_conf': 0.55,    # Backtest v2: 0% LONG (0/2), 61% SHORT, +66.49%
        'short_conf': 0.55,   # Backtest v2
        'long_meta_conf': 0.0,    # Disabled — not used in validated backtest
        'short_meta_conf': 0.0,   # Disabled
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
        'long_conf': 0.85,    # Backtest v2: 0 LONG trades, 62% SHORT, +50.84%
        'short_conf': 0.55,   # Backtest v2
        'long_meta_conf': 0.0,    # Disabled — not used in validated backtest
        'short_meta_conf': 0.0,   # Disabled
        'v3': True,
        'data_start': '2017-12-01',
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
        'long_conf': 0.70,    # Backtest v2: 50% LONG (8/16), 76% SHORT, +30.48%
        'short_conf': 0.52,   # Backtest v2
        'long_meta_conf': 0.0,    # Disabled
        'short_meta_conf': 0.0,   # Disabled
        'v3': True,
        'data_start': '2020-10-01',
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
        'long_conf': 0.55,    # Backtest v1: 71% LONG (10/14), 67% SHORT, +24.39%
        'short_conf': 0.55,   # Backtest v1
        'long_meta_conf': 0.0,    # Disabled
        'short_meta_conf': 0.0,   # Disabled
        'v3': True,
        'data_start': '2020-10-15',
    },
    'MATIC': {
        'pair': 'MATIC/USDT',
        'long_model': 'matic_cnn_model.pt',
        'short_model': 'matic_short_cnn_model.pt',
        'long_scaler': 'matic_feature_scaler.joblib',
        'short_scaler': 'matic_short_feature_scaler.joblib',
        'long_features': 'matic_features.json',
        'short_features': 'matic_short_features.json',
        'timeframes': ['4h', '1d', '1w'],
        'long_conf': 0.60,    # CNN confidence threshold
        'short_conf': 0.55,   # CNN SHORT confidence
        'long_meta_conf': 0.0,    # No meta model (pass-through)
        'short_meta_conf': 0.0,   # No meta model (pass-through)
        'v3': True,
        'data_start': '2021-05-01',
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
    HOST = '127.0.0.1'  # localhost only — nginx reverse proxy handles external HTTPS
    PORT = 8080
    DEBUG = os.getenv('API_DEBUG', 'false').lower() == 'true'
    CORS_ORIGINS = [
        'https://crypto-trading-bot.duckdns.org',
    ]
