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


def create_market_regime_features(df):
    """Market regime features: bull/bear/range detection, support/resistance, phase.
    Matches BTC_PRODUCTION/02_feature_engineering.py exactly."""
    sma_50 = df['close'].rolling(50).mean()
    sma_200 = df['close'].rolling(200).mean()
    df['sma_50'] = sma_50
    df['sma_200'] = sma_200
    df['sma50_above_sma200'] = (sma_50 > sma_200).astype(int)
    df['sma_spread_pct'] = (sma_50 / sma_200 - 1) * 100

    ema_20 = df['close'].rolling(20).mean()
    slope_20 = ema_20.pct_change(5) * 100
    df['regime_slope'] = slope_20
    df['regime_bull'] = ((slope_20 > 0.5) & (df['close'] > sma_50)).astype(int)
    df['regime_bear'] = ((slope_20 < -0.5) & (df['close'] < sma_50)).astype(int)
    df['regime_range'] = ((slope_20.abs() <= 0.5)).astype(int)

    high_20 = df['high'].rolling(20).max()
    low_20 = df['low'].rolling(20).min()
    df['resistance_dist_pct'] = (high_20 / df['close'] - 1) * 100
    df['support_dist_pct'] = (1 - low_20 / df['close']) * 100
    df['sr_range_pct'] = (high_20 - low_20) / df['close'] * 100

    high_50 = df['high'].rolling(50).max()
    low_50 = df['low'].rolling(50).min()
    df['resistance_50_dist_pct'] = (high_50 / df['close'] - 1) * 100
    df['support_50_dist_pct'] = (1 - low_50 / df['close']) * 100

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

    sma_cross = (sma_50 > sma_200).astype(int).diff()
    df['golden_cross_5d'] = sma_cross.rolling(5).sum().clip(0, 1)
    df['death_cross_5d'] = (-sma_cross).rolling(5).sum().clip(0, 1)

    rsi = df['1d_rsi_14'] if '1d_rsi_14' in df.columns else ta.momentum.RSIIndicator(df['close'], 14).rsi()
    df['rsi_regime_shift'] = rsi.diff(5)

    returns = df['close'].pct_change()
    vol_weighted_return = (returns * df['volume']).rolling(10).sum() / df['volume'].rolling(10).sum()
    df['vwap_trend_10'] = vol_weighted_return * 100

    df['buying_pressure'] = ((df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10))
    df['selling_pressure'] = ((df['high'] - df['close']) / (df['high'] - df['low'] + 1e-10))
    df['pressure_ratio'] = df['buying_pressure'] / (df['selling_pressure'] + 1e-10)

    df['trend_consistency_10'] = returns.rolling(10).apply(lambda x: (x > 0).sum() / len(x), raw=True)
    df['trend_consistency_20'] = returns.rolling(20).apply(lambda x: (x > 0).sum() / len(x), raw=True)

    # is_strong_bear for bear sample weighting context
    sma50_val = df['close'].rolling(50).mean()
    sma_dist = df['close'] / sma50_val - 1
    rsi_val = df['1d_rsi_14'] if '1d_rsi_14' in df.columns else ta.momentum.RSIIndicator(df['close'], 14).rsi()
    df['is_strong_bear'] = ((sma_dist < -0.10) | ((rsi_val < 30) & (sma_dist < -0.05))).astype(int)

    return df


def add_bear_features(df):
    """Bear-specific features for SHORT model.
    Matches BTC_PRODUCTION/03_train_short_model.py exactly."""
    for w in [10, 20, 50]:
        sma = df['close'].rolling(w).mean()
        df[f'price_above_sma{w}_pct'] = (df['close'] / sma - 1) * 100

    df['roc_5'] = df['close'].pct_change(5) * 100
    df['roc_10'] = df['close'].pct_change(10) * 100
    df['roc_deceleration'] = df['roc_5'] - df['roc_10']

    price_change_5 = df['close'].pct_change(5)
    vol_change_5 = df['volume'].pct_change(5)
    df['vol_price_divergence'] = np.where(
        (price_change_5 > 0) & (vol_change_5 < 0), 1,
        np.where((price_change_5 < 0) & (vol_change_5 > 0), -1, 0)
    )

    df['is_red'] = (df['close'] < df['open']).astype(int)
    df['consec_red'] = 0
    for i in range(1, len(df)):
        if df.iloc[i]['is_red']:
            df.iloc[i, df.columns.get_loc('consec_red')] = df.iloc[i-1]['consec_red'] + 1

    df['high_rejection'] = (df['high'] - df['close']) / (df['high'] - df['low'] + 1e-10)
    df['dist_from_high_20'] = (df['close'] / df['high'].rolling(20).max() - 1) * 100
    df['dist_from_high_50'] = (df['close'] / df['high'].rolling(50).max() - 1) * 100

    df['vol_expansion'] = df['1d_atr_14'] / df['1d_atr_14'].shift(5) if '1d_atr_14' in df.columns else 1

    if '1d_rsi_14' in df.columns:
        df['rsi_slope_5'] = df['1d_rsi_14'].diff(5)
        df['price_slope_5'] = df['close'].pct_change(5) * 100
        df['rsi_price_divergence'] = np.where(
            (df['price_slope_5'] > 0) & (df['rsi_slope_5'] < 0), 1,
            np.where((df['price_slope_5'] < 0) & (df['rsi_slope_5'] > 0), -1, 0)
        )

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

        # Market regime features (needed for BTC new model, harmless for others)
        has_regime = any(c in feature_cols for c in ['sma_50', 'regime_bull', 'accumulation_score'])
        if has_regime:
            df_1d = create_market_regime_features(df_1d)

        # Bear features (needed for BTC SHORT model)
        has_bear = any(c in feature_cols for c in ['price_above_sma10_pct', 'roc_5', 'consec_red'])
        if has_bear:
            df_1d = add_bear_features(df_1d)

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
