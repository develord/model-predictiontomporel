"""
BTC Feature Engineering Script - Multi-Timeframe + Non-Technical
================================================================
Creates enriched features:
- Multi-TF technical indicators (4h, 1d, 1w) like ETH/SOL
- Cross-timeframe alignment signals
- Non-technical features (volatility regimes, momentum coherence)
- Market microstructure features

Usage:
    python 02_feature_engineering.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
import ta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data' / 'cache'

CRYPTO = 'DOGE'
TP_PCT = 0.015
SL_PCT = 0.0075


def create_technical_indicators(df, prefix=''):
    """Create standard technical indicators for a timeframe"""
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

    # Additional: momentum and volatility
    df[f'{prefix}momentum_5'] = df['close'].pct_change(5)
    df[f'{prefix}momentum_10'] = df['close'].pct_change(10)
    df[f'{prefix}hist_vol_20'] = df['close'].pct_change().rolling(20).std()
    df[f'{prefix}volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()

    return df


def create_cross_tf_features(df):
    """Create cross-timeframe alignment and divergence features"""
    logger.info("  Creating cross-timeframe features...")

    # RSI alignment across timeframes
    for tf_pair in [('1d', '4h'), ('1d', '1w'), ('4h', '1w')]:
        tf1, tf2 = tf_pair
        col1 = f'{tf1}_rsi_14'
        col2 = f'{tf2}_rsi_14'
        if col1 in df.columns and col2 in df.columns:
            df[f'rsi_diff_{tf1}_{tf2}'] = df[col1] - df[col2]

    # RSI alignment score (all TFs bullish?)
    rsi_cols = [c for c in df.columns if c.endswith('_rsi_14')]
    if len(rsi_cols) >= 2:
        df['rsi_bullish_count'] = sum((df[c] > 50).astype(int) for c in rsi_cols)
        df['rsi_oversold_count'] = sum((df[c] < 30).astype(int) for c in rsi_cols)
        df['rsi_overbought_count'] = sum((df[c] > 70).astype(int) for c in rsi_cols)

    # MACD alignment
    macd_cols = [c for c in df.columns if c.endswith('_macd_histogram')]
    if len(macd_cols) >= 2:
        df['macd_bullish_count'] = sum((df[c] > 0).astype(int) for c in macd_cols)

    # Momentum alignment
    mom_cols = [c for c in df.columns if c.endswith('_momentum_5')]
    if len(mom_cols) >= 2:
        df['momentum_bullish_count'] = sum((df[c] > 0).astype(int) for c in mom_cols)

    # Trend strength consensus
    adx_cols = [c for c in df.columns if c.endswith('_adx_14')]
    if len(adx_cols) >= 2:
        df['adx_strong_count'] = sum((df[c] > 25).astype(int) for c in adx_cols)
        df['adx_mean'] = sum(df[c] for c in adx_cols) / len(adx_cols)

    # Volatility regime
    vol_cols = [c for c in df.columns if c.endswith('_hist_vol_20')]
    if len(vol_cols) >= 2:
        df['vol_mean_all_tf'] = sum(df[c] for c in vol_cols) / len(vol_cols)

    return df


def create_non_technical_features(df):
    """Create non-technical / market structure features derived from price/volume"""
    logger.info("  Creating non-technical features...")

    # Volatility regime (derived from price)
    df['daily_range_pct'] = (df['high'] - df['low']) / df['close']
    df['daily_range_ma5'] = df['daily_range_pct'].rolling(5).mean()
    df['daily_range_ma20'] = df['daily_range_pct'].rolling(20).mean()
    df['volatility_regime'] = (df['daily_range_ma5'] / df['daily_range_ma20']).fillna(1)

    # Volume profile
    df['volume_ma5'] = df['volume'].rolling(5).mean()
    df['volume_ma20'] = df['volume'].rolling(20).mean()
    df['volume_trend'] = (df['volume_ma5'] / df['volume_ma20']).fillna(1)

    # Price position relative to recent range
    df['price_position_20'] = (df['close'] - df['low'].rolling(20).min()) / \
                               (df['high'].rolling(20).max() - df['low'].rolling(20).min() + 1e-10)
    df['price_position_50'] = (df['close'] - df['low'].rolling(50).min()) / \
                               (df['high'].rolling(50).max() - df['low'].rolling(50).min() + 1e-10)

    # Candle patterns
    body = abs(df['close'] - df['open'])
    wick = df['high'] - df['low']
    df['body_ratio'] = body / (wick + 1e-10)
    df['upper_shadow_ratio'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (wick + 1e-10)
    df['lower_shadow_ratio'] = (df[['open', 'close']].min(axis=1) - df['low']) / (wick + 1e-10)

    # Consecutive direction
    df['returns'] = df['close'].pct_change()
    df['consecutive_up'] = 0
    df['consecutive_down'] = 0
    for i in range(1, len(df)):
        if df.iloc[i]['returns'] > 0:
            df.iloc[i, df.columns.get_loc('consecutive_up')] = df.iloc[i-1]['consecutive_up'] + 1
        if df.iloc[i]['returns'] < 0:
            df.iloc[i, df.columns.get_loc('consecutive_down')] = df.iloc[i-1]['consecutive_down'] + 1

    # Mean reversion signal
    df['distance_from_sma20'] = (df['close'] / df['close'].rolling(20).mean() - 1)
    df['distance_from_sma50'] = (df['close'] / df['close'].rolling(50).mean() - 1)

    # Trend strength
    df['higher_highs'] = (df['high'] > df['high'].shift(1)).rolling(5).sum()
    df['lower_lows'] = (df['low'] < df['low'].shift(1)).rolling(5).sum()
    df['trend_score'] = df['higher_highs'] - df['lower_lows']

    # Day of week / month
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_month'] = df['date'].dt.day
    df['month'] = df['date'].dt.month

    return df


def create_labels(df):
    """Create triple barrier labels"""
    labels = []
    for i in range(len(df)):
        if i >= len(df) - 1:
            labels.append(-1)
            continue

        entry = df.iloc[i]['close']
        tp = entry * (1 + TP_PCT)
        sl = entry * (1 - SL_PCT)

        hit_tp = False
        hit_sl = False

        for j in range(i+1, min(i+11, len(df))):
            if df.iloc[j]['high'] >= tp:
                hit_tp = True
                break
            if df.iloc[j]['low'] <= sl:
                hit_sl = True
                break

        if hit_tp:
            labels.append(1)
        elif hit_sl:
            labels.append(0)
        else:
            labels.append(-1)

    df['label'] = labels
    return df


def build_features():
    """Main pipeline: merge timeframes, create features, save"""
    logger.info(f"\n{'='*70}")
    logger.info(f"DOGE MULTI-TF FEATURE ENGINEERING")
    logger.info(f"{'='*70}\n")

    # Load 1d as base
    df_1d = pd.read_csv(DATA_DIR / f'{CRYPTO.lower()}_1d_data.csv')
    df_1d['date'] = pd.to_datetime(df_1d['date'])
    logger.info(f"Base 1d data: {len(df_1d)} candles")

    # Create 1d technical indicators
    df_1d = create_technical_indicators(df_1d, '1d_')

    # Load and merge other timeframes
    for tf in ['4h', '1w']:
        tf_file = DATA_DIR / f'{CRYPTO.lower()}_{tf}_data.csv'
        if not tf_file.exists():
            logger.warning(f"  {tf} data not found, skipping")
            continue

        df_tf = pd.read_csv(tf_file)
        df_tf['date'] = pd.to_datetime(df_tf['date'])
        df_tf = create_technical_indicators(df_tf, f'{tf}_')

        tf_cols = ['date'] + [c for c in df_tf.columns if c.startswith(f'{tf}_')]
        df_1d = pd.merge_asof(
            df_1d.sort_values('date'),
            df_tf[tf_cols].sort_values('date'),
            on='date',
            direction='backward'
        )
        logger.info(f"  Merged {tf}: {len(df_1d)} rows, +{len(tf_cols)-1} features")

    # Cross-TF features
    df_1d = create_cross_tf_features(df_1d)

    # Non-technical features
    df_1d = create_non_technical_features(df_1d)

    # Labels
    logger.info("  Creating labels...")
    df_1d = create_labels(df_1d)

    # Identify feature columns (exclude meta columns)
    exclude = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'date', 'label',
               'label_class', 'triple_barrier_label', 'returns']
    feature_cols = [c for c in df_1d.columns if c not in exclude and not c.startswith('Unnamed')]

    # Clean inf/nan in features
    for c in feature_cols:
        df_1d[c] = pd.to_numeric(df_1d[c], errors='coerce')
    df_1d[feature_cols] = df_1d[feature_cols].replace([np.inf, -np.inf], np.nan)

    # Save features list
    with open(BASE_DIR / 'required_features.json', 'w') as f:
        json.dump(feature_cols, f, indent=2)

    # Save merged data
    output_file = DATA_DIR / 'doge_features.csv'
    df_1d.to_csv(output_file, index=False)

    # Stats
    n_labeled = (df_1d['label'] != -1).sum()
    n_tp = (df_1d['label'] == 1).sum()
    n_sl = (df_1d['label'] == 0).sum()

    logger.info(f"\n{'='*70}")
    logger.info(f"FEATURE ENGINEERING COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"  Total rows: {len(df_1d)}")
    logger.info(f"  Total features: {len(feature_cols)}")
    logger.info(f"  Labeled: {n_labeled} (TP={n_tp}, SL={n_sl})")
    logger.info(f"  Saved to {output_file}")
    logger.info(f"  Features list saved to required_features.json")

    return df_1d


if __name__ == "__main__":
    build_features()
