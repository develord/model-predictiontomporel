"""
Feature Engine - Compute features identical to training
========================================================
Uses same ta library and logic as 02_feature_engineering.py
"""

import pandas as pd
import numpy as np
import ta
import logging

logger = logging.getLogger(__name__)


def create_technical_indicators(df, prefix=''):
    """Create technical indicators for a timeframe - matches training exactly"""
    df = df.copy()

    df[f'{prefix}rsi_14'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    df[f'{prefix}rsi_21'] = ta.momentum.RSIIndicator(df['close'], window=21).rsi()

    macd = ta.trend.MACD(df['close'])
    df[f'{prefix}macd_line'] = macd.macd()
    df[f'{prefix}macd_signal'] = macd.macd_signal()
    df[f'{prefix}macd_histogram'] = macd.macd_diff()

    bb = ta.volatility.BollingerBands(df['close'])
    df[f'{prefix}bb_upper'] = bb.bollinger_hband()
    df[f'{prefix}bb_middle'] = bb.bollinger_mavg()
    df[f'{prefix}bb_lower'] = bb.bollinger_lband()
    df[f'{prefix}bb_width'] = bb.bollinger_wband()

    df[f'{prefix}ema_12'] = ta.trend.EMAIndicator(df['close'], window=12).ema_indicator()
    df[f'{prefix}ema_26'] = ta.trend.EMAIndicator(df['close'], window=26).ema_indicator()
    df[f'{prefix}ema_50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()

    df[f'{prefix}atr_14'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()

    stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
    df[f'{prefix}stoch_k'] = stoch.stoch()
    df[f'{prefix}stoch_d'] = stoch.stoch_signal()

    df[f'{prefix}adx_14'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()

    df[f'{prefix}obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
    df[f'{prefix}cmf_20'] = ta.volume.ChaikinMoneyFlowIndicator(df['high'], df['low'], df['close'], df['volume'], window=20).chaikin_money_flow()

    df[f'{prefix}momentum_5'] = df['close'].pct_change(5)
    df[f'{prefix}momentum_10'] = df['close'].pct_change(10)
    df[f'{prefix}hist_vol_20'] = df['close'].pct_change().rolling(20).std()
    df[f'{prefix}volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()

    return df


def create_cross_tf_features(df):
    """Cross-timeframe alignment features"""
    for tf1, tf2 in [('1d', '4h'), ('1d', '1w'), ('4h', '1w')]:
        c1, c2 = f'{tf1}_rsi_14', f'{tf2}_rsi_14'
        if c1 in df.columns and c2 in df.columns:
            df[f'rsi_diff_{tf1}_{tf2}'] = df[c1] - df[c2]

    rsi_cols = [c for c in df.columns if c.endswith('_rsi_14')]
    if len(rsi_cols) >= 2:
        df['rsi_bullish_count'] = sum((df[c] > 50).astype(int) for c in rsi_cols)
        df['rsi_oversold_count'] = sum((df[c] < 30).astype(int) for c in rsi_cols)
        df['rsi_overbought_count'] = sum((df[c] > 70).astype(int) for c in rsi_cols)

    macd_cols = [c for c in df.columns if c.endswith('_macd_histogram')]
    if len(macd_cols) >= 2:
        df['macd_bullish_count'] = sum((df[c] > 0).astype(int) for c in macd_cols)

    mom_cols = [c for c in df.columns if c.endswith('_momentum_5')]
    if len(mom_cols) >= 2:
        df['momentum_bullish_count'] = sum((df[c] > 0).astype(int) for c in mom_cols)

    adx_cols = [c for c in df.columns if c.endswith('_adx_14')]
    if len(adx_cols) >= 2:
        df['adx_strong_count'] = sum((df[c] > 25).astype(int) for c in adx_cols)
        df['adx_mean'] = sum(df[c] for c in adx_cols) / len(adx_cols)

    vol_cols = [c for c in df.columns if c.endswith('_hist_vol_20')]
    if len(vol_cols) >= 2:
        df['vol_mean_all_tf'] = sum(df[c] for c in vol_cols) / len(vol_cols)

    return df


def create_non_technical_features(df):
    """Non-technical / market structure features"""
    df['daily_range_pct'] = (df['high'] - df['low']) / df['close']
    df['daily_range_ma5'] = df['daily_range_pct'].rolling(5).mean()
    df['daily_range_ma20'] = df['daily_range_pct'].rolling(20).mean()
    df['volatility_regime'] = (df['daily_range_ma5'] / df['daily_range_ma20']).fillna(1)

    df['volume_ma5'] = df['volume'].rolling(5).mean()
    df['volume_ma20'] = df['volume'].rolling(20).mean()
    df['volume_trend'] = (df['volume_ma5'] / df['volume_ma20']).fillna(1)

    df['price_position_20'] = (df['close'] - df['low'].rolling(20).min()) / \
                               (df['high'].rolling(20).max() - df['low'].rolling(20).min() + 1e-10)
    df['price_position_50'] = (df['close'] - df['low'].rolling(50).min()) / \
                               (df['high'].rolling(50).max() - df['low'].rolling(50).min() + 1e-10)

    body = abs(df['close'] - df['open'])
    wick = df['high'] - df['low']
    df['body_ratio'] = body / (wick + 1e-10)
    df['upper_shadow_ratio'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (wick + 1e-10)
    df['lower_shadow_ratio'] = (df[['open', 'close']].min(axis=1) - df['low']) / (wick + 1e-10)

    df['returns'] = df['close'].pct_change()
    df['consecutive_up'] = 0
    df['consecutive_down'] = 0
    for i in range(1, len(df)):
        if df.iloc[i]['returns'] > 0:
            df.iloc[i, df.columns.get_loc('consecutive_up')] = df.iloc[i-1]['consecutive_up'] + 1
        if df.iloc[i]['returns'] < 0:
            df.iloc[i, df.columns.get_loc('consecutive_down')] = df.iloc[i-1]['consecutive_down'] + 1

    df['distance_from_sma20'] = (df['close'] / df['close'].rolling(20).mean() - 1)
    df['distance_from_sma50'] = (df['close'] / df['close'].rolling(50).mean() - 1)

    df['higher_highs'] = (df['high'] > df['high'].shift(1)).rolling(5).sum()
    df['lower_lows'] = (df['low'] < df['low'].shift(1)).rolling(5).sum()
    df['trend_score'] = df['higher_highs'] - df['lower_lows']

    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_month'] = df['date'].dt.day
    df['month'] = df['date'].dt.month

    return df


def compute_features(buffers, coin, feature_cols, scaler, seq_len=30):
    """
    Compute features from OHLCV buffers for a coin.
    Returns scaled feature array of shape (seq_len, feature_dim) or None.
    """
    try:
        df_1d = buffers[coin]['1d'].copy()
        if len(df_1d) < seq_len + 60:
            logger.warning(f"{coin}: Not enough 1d data ({len(df_1d)} rows)")
            return None, None

        if 'date' not in df_1d.columns:
            df_1d['date'] = pd.to_datetime(df_1d['timestamp'], unit='ms') if 'timestamp' in df_1d.columns else df_1d.index

        # Create 1d indicators
        df_1d = create_technical_indicators(df_1d, '1d_')

        # Merge other timeframes
        for tf in ['4h', '1w']:
            if tf in buffers[coin] and len(buffers[coin][tf]) > 0:
                df_tf = buffers[coin][tf].copy()
                if 'date' not in df_tf.columns:
                    df_tf['date'] = pd.to_datetime(df_tf['timestamp'], unit='ms') if 'timestamp' in df_tf.columns else df_tf.index
                df_tf = create_technical_indicators(df_tf, f'{tf}_')

                tf_cols = ['date'] + [c for c in df_tf.columns if c.startswith(f'{tf}_')]
                df_1d = pd.merge_asof(
                    df_1d.sort_values('date'),
                    df_tf[tf_cols].sort_values('date'),
                    on='date',
                    direction='backward'
                )

        # Cross-TF + non-technical
        df_1d = create_cross_tf_features(df_1d)
        df_1d = create_non_technical_features(df_1d)

        # Clean
        for c in feature_cols:
            if c not in df_1d.columns:
                df_1d[c] = 0
            df_1d[c] = pd.to_numeric(df_1d[c], errors='coerce')
        df_1d[feature_cols] = df_1d[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

        # Extract last seq_len rows (raw for filters)
        raw_row = df_1d.iloc[-1]

        # Scale
        feat_values = df_1d[feature_cols].values.astype(np.float32)
        scaled = np.clip(np.nan_to_num(scaler.transform(feat_values), nan=0, posinf=0, neginf=0), -5, 5)

        # Return last seq_len rows
        if len(scaled) < seq_len:
            logger.warning(f"{coin}: Not enough scaled data ({len(scaled)} < {seq_len})")
            return None, None

        return scaled[-seq_len:], raw_row

    except Exception as e:
        logger.error(f"{coin} feature computation error: {e}")
        return None, None
