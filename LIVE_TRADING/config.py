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
        'long_conf': 0.60,
        'short_conf': 0.55,   # Backtest original optimized
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
        'long_conf': 0.60,
        'short_conf': 0.55,   # Backtest original optimized
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
        'long_conf': 0.55,
        'short_conf': 0.68,   # Backtest original: très sélectif SHORT
        'bear_sma50': -0.12,  # Per-coin override (backtest original)
        'bear_sma20': -0.05,  # Per-coin override (backtest original)
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
        'long_conf': 0.85,    # Backtest original: très sélectif LONG
        'short_conf': 0.55,   # Backtest original optimized
        'bear_sma50': -0.12,  # Per-coin override (backtest original)
        'bear_sma20': -0.05,  # Per-coin override (backtest original)
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
        'long_conf': 0.70,
        'short_conf': 0.52,   # Backtest original optimized
        'bear_sma50': -0.12,  # Per-coin override (backtest original)
        'bear_sma20': -0.05,  # Per-coin override (backtest original)
        'data_start': '2020-10-01',
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
    'max_consecutive_losses': 3,
    'cooldown_days': 5,
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
