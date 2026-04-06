"""
NEAR Feature Engineering Script - Multi-Timeframe + Non-Technical
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

CRYPTO = 'NEAR'
# ATR-based labeling params (used as fallback multipliers)
ATR_TP_MULT = 1.5   # TP = 1.5x ATR (balanced with SL)
ATR_SL_MULT = 1.5   # SL = 1.5x ATR (symmetric = no directional bias)
FIXED_TP_PCT = 0.012  # Fallback: 1.2% symmetric
FIXED_SL_PCT = 0.012  # Fallback: 1.2% symmetric
BASE_LOOKAHEAD = 10
ATR_LOOKAHEAD_MULT = 0.7  # Adaptive: low vol = shorter, high vol = longer


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


def create_market_regime_features(df):
    """Create market regime features: bull/bear/range detection, support/resistance, phase"""
    logger.info("  Creating market regime features...")

    # --- Trend regime via SMA crossovers ---
    sma_50 = df['close'].rolling(50).mean()
    sma_200 = df['close'].rolling(200).mean()
    df['sma_50'] = sma_50
    df['sma_200'] = sma_200
    df['sma50_above_sma200'] = (sma_50 > sma_200).astype(int)
    df['sma_spread_pct'] = (sma_50 / sma_200 - 1) * 100  # Distance between SMAs

    # Bull/Bear/Range classifier (3-state)
    ema_20 = df['close'].rolling(20).mean()
    slope_20 = ema_20.pct_change(5) * 100  # 5-day slope of EMA20
    df['regime_slope'] = slope_20
    df['regime_bull'] = ((slope_20 > 0.5) & (df['close'] > sma_50)).astype(int)
    df['regime_bear'] = ((slope_20 < -0.5) & (df['close'] < sma_50)).astype(int)
    df['regime_range'] = ((slope_20.abs() <= 0.5)).astype(int)

    # --- Support/Resistance via rolling pivot points ---
    high_20 = df['high'].rolling(20).max()
    low_20 = df['low'].rolling(20).min()
    df['resistance_dist_pct'] = (high_20 / df['close'] - 1) * 100
    df['support_dist_pct'] = (1 - low_20 / df['close']) * 100
    df['sr_range_pct'] = (high_20 - low_20) / df['close'] * 100  # Range width

    high_50 = df['high'].rolling(50).max()
    low_50 = df['low'].rolling(50).min()
    df['resistance_50_dist_pct'] = (high_50 / df['close'] - 1) * 100
    df['support_50_dist_pct'] = (1 - low_50 / df['close']) * 100

    # --- Market phase (Wyckoff-inspired) ---
    vol_trend = df['volume'].rolling(10).mean() / df['volume'].rolling(30).mean()
    price_pos = (df['close'] - low_20) / (high_20 - low_20 + 1e-10)
    df['volume_trend_ratio'] = vol_trend
    df['price_position_in_range'] = price_pos

    df['accumulation_score'] = (
        (df['regime_range'] == 1).astype(float) *
        (vol_trend > 1).astype(float) *
        (price_pos < 0.3).astype(float)
    )
    df['distribution_score'] = (
        (df['regime_range'] == 1).astype(float) *
        (vol_trend > 1).astype(float) *
        (price_pos > 0.7).astype(float)
    )

    # --- Regime change detection ---
    sma_cross = (sma_50 > sma_200).astype(int).diff()
    df['golden_cross_5d'] = sma_cross.rolling(5).sum().clip(0, 1)
    df['death_cross_5d'] = (-sma_cross).rolling(5).sum().clip(0, 1)

    # Momentum regime shift
    rsi = df['1d_rsi_14'] if '1d_rsi_14' in df.columns else ta.momentum.RSIIndicator(df['close'], 14).rsi()
    df['rsi_regime_shift'] = rsi.diff(5)  # RSI change over 5 bars

    # --- Volume-weighted trend strength ---
    returns = df['close'].pct_change()
    vol_weighted_return = (returns * df['volume']).rolling(10).sum() / df['volume'].rolling(10).sum()
    df['vwap_trend_10'] = vol_weighted_return * 100

    # Buying vs selling pressure
    df['buying_pressure'] = ((df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10))
    df['selling_pressure'] = ((df['high'] - df['close']) / (df['high'] - df['low'] + 1e-10))
    df['pressure_ratio'] = df['buying_pressure'] / (df['selling_pressure'] + 1e-10)

    # --- Trend persistence ---
    df['trend_consistency_10'] = returns.rolling(10).apply(lambda x: (x > 0).sum() / len(x), raw=True)
    df['trend_consistency_20'] = returns.rolling(20).apply(lambda x: (x > 0).sum() / len(x), raw=True)

    return df


def create_labels(df):
    """Create regime-aware ATR-adaptive triple barrier labels."""
    # Compute ATR for adaptive thresholds
    atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
    median_atr = atr.rolling(50).median()

    # Regime detection for label suppression
    sma_50 = df['close'].rolling(50).mean()
    sma_dist = (df['close'] / sma_50 - 1)  # Negative = below SMA50
    rsi = ta.momentum.RSIIndicator(df['close'], window=14).rsi()

    labels = []
    n_suppressed = 0
    for i in range(len(df)):
        if i >= len(df) - 1:
            labels.append(-1)
            continue

        entry = df.iloc[i]['close']
        current_atr = atr.iloc[i]

        # ATR-based TP/SL (symmetric)
        if pd.notna(current_atr) and current_atr > 0:
            tp_dist = current_atr * ATR_TP_MULT
            sl_dist = current_atr * ATR_SL_MULT
        else:
            tp_dist = entry * FIXED_TP_PCT
            sl_dist = entry * FIXED_SL_PCT

        tp = entry + tp_dist
        sl = entry - sl_dist

        # Adaptive lookahead
        med = median_atr.iloc[i] if pd.notna(median_atr.iloc[i]) and median_atr.iloc[i] > 0 else current_atr
        if pd.notna(med) and med > 0:
            vol_ratio = current_atr / med
            lookahead = int(BASE_LOOKAHEAD * max(0.5, min(2.0, vol_ratio * ATR_LOOKAHEAD_MULT + 0.3)))
        else:
            lookahead = BASE_LOOKAHEAD
        lookahead = max(5, min(20, lookahead))

        hit_tp = False
        hit_sl = False

        for j in range(i + 1, min(i + 1 + lookahead, len(df))):
            if df.iloc[j]['high'] >= tp:
                hit_tp = True
                break
            if df.iloc[j]['low'] <= sl:
                hit_sl = True
                break

        cur_sma_dist = sma_dist.iloc[i] if pd.notna(sma_dist.iloc[i]) else 0
        cur_rsi = rsi.iloc[i] if pd.notna(rsi.iloc[i]) else 50

        is_strong_bear = (cur_sma_dist < -0.10) or (cur_rsi < 30 and cur_sma_dist < -0.05)

        if hit_tp and is_strong_bear:
            labels.append(1)
            n_suppressed += 1
        elif hit_tp:
            labels.append(1)
        elif hit_sl:
            labels.append(0)
        else:
            labels.append(-1)

    df['label'] = labels

    df['is_strong_bear'] = (
        (sma_dist < -0.10) | ((rsi < 30) & (sma_dist < -0.05))
    ).astype(int)

    logger.info(f"  Labels: TP in bear market (downweighted): {n_suppressed}")
    logger.info(f"  Strong bear days: {df['is_strong_bear'].sum()}")
    return df


def build_features():
    """Main pipeline: merge timeframes, create features, save"""
    logger.info(f"\n{'='*70}")
    logger.info(f"NEAR MULTI-TF FEATURE ENGINEERING")
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

    # Market regime features
    df_1d = create_market_regime_features(df_1d)

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
    output_file = DATA_DIR / 'near_features.csv'
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
