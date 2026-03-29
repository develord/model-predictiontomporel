"""
SOL Feature Engineering Script
===============================
Generates multi-timeframe features (1h, 4h, 1d, 1w) and creates labels

Usage:
    python 02_feature_engineering.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import ta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data' / 'cache'

CRYPTO = 'SOL'
TP_PCT = 0.015  # 1.5%
SL_PCT = 0.0075  # 0.75%


def create_technical_indicators(df, prefix=''):
    """Create technical indicators for given timeframe"""
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


def create_triple_barrier_labels(df):
    """Create triple barrier labels (TP/SL)"""
    df['label_class'] = 0  # Default: no trade

    for i in range(len(df) - 1):
        entry_price = df.iloc[i]['close']
        tp_price = entry_price * (1 + TP_PCT)
        sl_price = entry_price * (1 - SL_PCT)

        # Check future candles
        future = df.iloc[i+1:]
        tp_hit = (future['high'] >= tp_price).any()
        sl_hit = (future['low'] <= sl_price).any()

        if tp_hit and not sl_hit:
            df.at[df.index[i], 'label_class'] = 1  # TP hit
        elif sl_hit:
            df.at[df.index[i], 'label_class'] = -1  # SL hit

    # Remap for XGBoost: -1 → 0
    df['triple_barrier_label'] = df['label_class'].replace(-1, 0)

    return df


def merge_timeframes():
    """Merge all timeframes into single dataset"""
    logger.info(f"\n{'='*70}")
    logger.info(f"MERGING {CRYPTO} TIMEFRAMES")
    logger.info(f"{'='*70}\n")

    # Load 1d as base
    df_1d = pd.read_csv(DATA_DIR / f'{CRYPTO.lower()}_1d_data.csv')
    df_1d['date'] = pd.to_datetime(df_1d['date'])

    logger.info(f"Base 1d data: {len(df_1d)} candles")

    # Create features for 1d
    df_1d = create_technical_indicators(df_1d, '1d_')

    # Load and merge other timeframes
    for tf in ['1h', '4h', '1w']:
        df_tf = pd.read_csv(DATA_DIR / f'{CRYPTO.lower()}_{tf}_data.csv')
        df_tf['date'] = pd.to_datetime(df_tf['date'])
        df_tf = create_technical_indicators(df_tf, f'{tf}_')

        # Merge on date
        df_1d = pd.merge_asof(
            df_1d.sort_values('date'),
            df_tf[['date'] + [col for col in df_tf.columns if col.startswith(tf)]].sort_values('date'),
            on='date',
            direction='backward'
        )

        logger.info(f"Merged {tf}: {len(df_1d)} candles")

    # Create labels
    df_1d = create_triple_barrier_labels(df_1d)

    # Save
    output_file = DATA_DIR / f'{CRYPTO.lower()}_multi_tf_merged.csv'
    df_1d.to_csv(output_file, index=False)

    logger.info(f"\n✓ Saved merged data to {output_file}")
    logger.info(f"  Total features: {len(df_1d.columns)}")
    logger.info(f"  Total candles: {len(df_1d)}")
    logger.info(f"  Label distribution:")
    logger.info(f"    TP (1): {(df_1d['triple_barrier_label'] == 1).sum()}")
    logger.info(f"    SL/None (0): {(df_1d['triple_barrier_label'] == 0).sum()}")

    return df_1d


if __name__ == "__main__":
    merge_timeframes()
