"""
SOL Production Inference Script
================================
Real-time predictions using trained XGBoost model with intelligent filtering

Usage:
    python 05_production_inference.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import joblib
import json
import ccxt
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data' / 'cache'
MODEL_DIR = BASE_DIR / 'models'

CRYPTO = 'SOL'
TIMEFRAMES = ['1h', '4h', '1d', '1w']

# Filtering thresholds
MIN_CONFIDENCE = 0.65
MAX_VOLATILITY_1D = 0.04
MAX_VOLATILITY_4H = 0.03
MAX_VOLATILITY_1W = 0.05
MIN_VOLUME_RATIO = 1.2
MIN_ADX = 20
MIN_MOMENTUM_ALIGNMENT = 2


def fetch_latest_data():
    """Fetch latest OHLCV data from Binance"""
    logger.info(f"Fetching latest {CRYPTO} data from Binance...")

    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'}
    })

    symbol = f'{CRYPTO}/USDT'
    data = {}

    for tf in TIMEFRAMES:
        ohlcv = exchange.fetch_ohlcv(symbol, tf, limit=300)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        data[tf] = df
        logger.info(f"  ✓ {tf}: {len(df)} candles")

    return data


def create_technical_indicators(df, prefix=''):
    """Create technical indicators (same as feature engineering)"""
    import ta

    # RSI
    df[f'{prefix}rsi_14'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    df[f'{prefix}rsi_21'] = ta.momentum.RSIIndicator(df['close'], window=21).rsi()

    # MACD
    macd = ta.trend.MACD(df['close'])
    df[f'{prefix}macd_line'] = macd.macd()
    df[f'{prefix}macd_signal'] = macd.macd_signal()
    df[f'{prefix}macd_histogram'] = macd.macd_diff()

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df['close'])
    df[f'{prefix}bb_upper'] = bb.bollinger_hband()
    df[f'{prefix}bb_middle'] = bb.bollinger_mavg()
    df[f'{prefix}bb_lower'] = bb.bollinger_lband()
    df[f'{prefix}bb_width'] = bb.bollinger_wband()

    # EMAs
    df[f'{prefix}ema_12'] = ta.trend.EMAIndicator(df['close'], window=12).ema_indicator()
    df[f'{prefix}ema_26'] = ta.trend.EMAIndicator(df['close'], window=26).ema_indicator()
    df[f'{prefix}ema_50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
    df[f'{prefix}ema_200'] = ta.trend.EMAIndicator(df['close'], window=200).ema_indicator()

    # ATR
    df[f'{prefix}atr_14'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()

    # Stochastic
    stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
    df[f'{prefix}stoch_k'] = stoch.stoch()
    df[f'{prefix}stoch_d'] = stoch.stoch_signal()

    # ADX
    df[f'{prefix}adx_14'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()

    # Volume indicators
    df[f'{prefix}obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
    df[f'{prefix}cmf_20'] = ta.volume.ChaikinMoneyFlowIndicator(df['high'], df['low'], df['close'], df['volume'], window=20).chaikin_money_flow()

    return df


def merge_features(data):
    """Merge all timeframe features"""
    # Use 1d as base
    df_1d = data['1d'].copy()
    df_1d = create_technical_indicators(df_1d, '1d_')

    # Merge other timeframes
    for tf in ['1h', '4h', '1w']:
        df_tf = data[tf].copy()
        df_tf = create_technical_indicators(df_tf, f'{tf}_')

        # Merge on date
        df_1d = pd.merge_asof(
            df_1d.sort_values('date'),
            df_tf[['date'] + [col for col in df_tf.columns if col.startswith(tf)]].sort_values('date'),
            on='date',
            direction='backward'
        )

    return df_1d


def check_momentum_alignment(row):
    """Check if momentum aligns across timeframes"""
    alignments = 0
    if row.get('1d_rsi_14', 50) > 50:
        alignments += 1
    if row.get('4h_rsi_14', 50) > 50:
        alignments += 1
    if row.get('1h_rsi_14', 50) > 50:
        alignments += 1
    return alignments


def intelligent_signal_filter(row, df):
    """5-criteria intelligent filtering system"""
    filters_passed = []
    filters_failed = []

    # 1. Confidence threshold
    if row['confidence'] >= MIN_CONFIDENCE:
        filters_passed.append(f"Confidence: {row['confidence']:.2%} ≥ {MIN_CONFIDENCE:.0%}")
    else:
        filters_failed.append(f"Confidence: {row['confidence']:.2%} < {MIN_CONFIDENCE:.0%}")
        return False, filters_passed, filters_failed

    # 2. Volatility checks
    vol_1d = df['close'].pct_change().tail(20).std()
    if vol_1d <= MAX_VOLATILITY_1D:
        filters_passed.append(f"Volatility 1d: {vol_1d:.2%} ≤ {MAX_VOLATILITY_1D:.0%}")
    else:
        filters_failed.append(f"Volatility 1d: {vol_1d:.2%} > {MAX_VOLATILITY_1D:.0%}")
        return False, filters_passed, filters_failed

    # 3. Volume filter
    avg_volume = df['volume'].tail(20).mean()
    volume_ratio = row['volume'] / avg_volume
    if volume_ratio >= MIN_VOLUME_RATIO:
        filters_passed.append(f"Volume ratio: {volume_ratio:.2f} ≥ {MIN_VOLUME_RATIO}")
    else:
        filters_failed.append(f"Volume ratio: {volume_ratio:.2f} < {MIN_VOLUME_RATIO}")
        return False, filters_passed, filters_failed

    # 4. Trend strength (ADX)
    if '1d_adx_14' in row and pd.notna(row['1d_adx_14']):
        if row['1d_adx_14'] >= MIN_ADX:
            filters_passed.append(f"ADX: {row['1d_adx_14']:.1f} ≥ {MIN_ADX}")
        else:
            filters_failed.append(f"ADX: {row['1d_adx_14']:.1f} < {MIN_ADX}")
            return False, filters_passed, filters_failed

    # 5. Multi-timeframe momentum alignment
    momentum_align = check_momentum_alignment(row)
    if momentum_align >= MIN_MOMENTUM_ALIGNMENT:
        filters_passed.append(f"Momentum alignment: {momentum_align}/3 ≥ {MIN_MOMENTUM_ALIGNMENT}")
    else:
        filters_failed.append(f"Momentum alignment: {momentum_align}/3 < {MIN_MOMENTUM_ALIGNMENT}")
        return False, filters_passed, filters_failed

    return True, filters_passed, filters_failed


def make_prediction():
    """Make real-time prediction"""
    logger.info(f"\n{'='*70}")
    logger.info(f"{CRYPTO} PRODUCTION INFERENCE")
    logger.info(f"{'='*70}\n")

    # Load model
    model = joblib.load(MODEL_DIR / f'{CRYPTO.lower()}_v11_top50.joblib')
    with open(MODEL_DIR / f'{CRYPTO.lower()}_v11_features.json', 'r') as f:
        feature_cols = json.load(f)

    logger.info(f"✓ Loaded model with {len(feature_cols)} features")

    # Fetch latest data
    data = fetch_latest_data()

    # Merge features
    df_merged = merge_features(data)

    # Get latest row
    latest = df_merged.iloc[-1]

    # Prepare features
    X = pd.DataFrame([latest[feature_cols]]).fillna(0)

    # Predict
    prediction = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    confidence = proba[1]  # Confidence for class 1 (TP)

    # Get current price
    current_price = latest['close']
    current_time = latest['date']

    logger.info(f"\n{'='*70}")
    logger.info(f"PREDICTION RESULT")
    logger.info(f"{'='*70}\n")
    logger.info(f"Time: {current_time}")
    logger.info(f"Current Price: ${current_price:.2f}")
    logger.info(f"Raw Prediction: {'LONG (TP Expected)' if prediction == 1 else 'NO TRADE'}")
    logger.info(f"Confidence: {confidence:.2%}")

    # Apply intelligent filtering
    if prediction == 1:
        passes_filter, filters_passed, filters_failed = intelligent_signal_filter(latest, df_merged)

        logger.info(f"\n{'='*70}")
        logger.info(f"INTELLIGENT FILTERING")
        logger.info(f"{'='*70}\n")

        logger.info(f"Filters Passed ({len(filters_passed)}):")
        for f in filters_passed:
            logger.info(f"  ✓ {f}")

        if filters_failed:
            logger.info(f"\nFilters Failed ({len(filters_failed)}):")
            for f in filters_failed:
                logger.info(f"  ✗ {f}")

        logger.info(f"\n{'='*70}")
        logger.info(f"FINAL DECISION")
        logger.info(f"{'='*70}\n")

        if passes_filter:
            tp_price = current_price * 1.015
            sl_price = current_price * 0.9925

            logger.info(f"✅ TRADE SIGNAL: ENTER LONG")
            logger.info(f"   Entry: ${current_price:.2f}")
            logger.info(f"   TP (1.5%): ${tp_price:.2f}")
            logger.info(f"   SL (0.75%): ${sl_price:.2f}")
            logger.info(f"   Confidence: {confidence:.2%}")
        else:
            logger.info(f"⛔ NO TRADE - Signal filtered out")
    else:
        logger.info(f"\n⛔ NO TRADE - Model predicts no opportunity")

    logger.info(f"\n{'='*70}")


if __name__ == "__main__":
    make_prediction()
