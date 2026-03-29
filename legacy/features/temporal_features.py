"""
V10 Temporal Features - Multi-Timeframe
========================================
Port of V8 temporal features with multi-timeframe support

Calculates 49 temporal features per timeframe:
- Lag features (5 lags): 10 features (5 price + 5 volume)
- Rolling statistics (4 windows): 12 features (3 per window)
- Momentum sequence (5 periods): 9 features (5 momentum + 4 acceleration)
- Price sequence pattern: 4 features
- Volume momentum (3 periods): 6 features
- Indicator momentum: 6 features
- Temporal correlation: 2 features

V8 Impact: Reduced overfitting by -7% while maintaining ROI
"""

import numpy as np
import pandas as pd
from typing import Dict, List


def calculate_lag_features(df: pd.DataFrame, timeframe: str, n_lags: int = 5) -> pd.DataFrame:
    """
    Lag features: previous N periods prices and volumes

    Args:
        df: DataFrame with 'close' and 'volume'
        timeframe: Timeframe identifier ('4h', '1d', '1w')
        n_lags: Number of lags (default 5)

    Returns:
        DataFrame with 10 lag features (5 price + 5 volume)
    """
    result = df.copy()
    tf = timeframe

    # Price lags (normalized as % change from current)
    for i in range(1, n_lags + 1):
        result[f'{tf}_close_lag_{i}'] = df['close'].pct_change(i) * 100

    # Volume lags (normalized as ratio to current)
    for i in range(1, n_lags + 1):
        result[f'{tf}_volume_lag_{i}'] = df['volume'] / df['volume'].shift(i)

    return result


def calculate_rolling_statistics(df: pd.DataFrame, timeframe: str, windows: List[int] = [3, 5, 7, 14]) -> pd.DataFrame:
    """
    Rolling statistics on multiple time windows

    Args:
        df: DataFrame with 'close'
        timeframe: Timeframe identifier
        windows: List of window sizes

    Returns:
        DataFrame with 12 features (3 per window: mean, volatility, trend)
    """
    result = df.copy()
    tf = timeframe

    for window in windows:
        # Rolling mean (price vs mean)
        rolling_mean = df['close'].rolling(window=window).mean()
        result[f'{tf}_price_vs_mean_{window}'] = ((df['close'] - rolling_mean) / rolling_mean * 100).fillna(0)

        # Rolling volatility
        rolling_std = df['close'].rolling(window=window).std()
        result[f'{tf}_volatility_{window}'] = (rolling_std / rolling_mean * 100).fillna(0)

        # Linear trend (slope)
        def calc_slope(x):
            if len(x) < 2:
                return 0
            y = np.array(x)
            slope = np.polyfit(np.arange(len(y)), y, 1)[0]
            return (slope / np.mean(y) * 100) if np.mean(y) > 0 else 0

        result[f'{tf}_trend_{window}'] = df['close'].rolling(window=window).apply(calc_slope, raw=True).fillna(0)

    return result


def calculate_momentum_sequence(df: pd.DataFrame, timeframe: str, periods: List[int] = [1, 3, 5, 7, 14]) -> pd.DataFrame:
    """
    Momentum over multiple periods to capture acceleration/deceleration

    Args:
        df: DataFrame with 'close'
        timeframe: Timeframe identifier
        periods: List of momentum periods

    Returns:
        DataFrame with 9 features (5 momentum + 4 acceleration)
    """
    result = df.copy()
    tf = timeframe

    # Individual momentums
    momentums = {}
    for period in periods:
        momentum = df['close'].pct_change(period) * 100
        result[f'{tf}_momentum_{period}'] = momentum.fillna(0)
        momentums[period] = momentum

    # Momentum acceleration (change in momentum)
    for i in range(len(periods) - 1):
        p1 = periods[i]
        p2 = periods[i + 1]
        result[f'{tf}_accel_{p1}_{p2}'] = (momentums[p1] - momentums[p2]).fillna(0)

    return result


def calculate_price_sequence_pattern(df: pd.DataFrame, timeframe: str, seq_length: int = 5) -> pd.DataFrame:
    """
    Encode recent price sequence as pattern features

    Args:
        df: DataFrame with 'close'
        timeframe: Timeframe identifier
        seq_length: Sequence length (default 5)

    Returns:
        DataFrame with 4 pattern features
    """
    result = df.copy()
    tf = timeframe

    # Higher highs / lower lows pattern
    def calc_higher_highs(x):
        if len(x) < 2:
            return 0
        return sum(1 for i in range(1, len(x)) if x[i] > x[i-1]) / (len(x) - 1)

    def calc_lower_lows(x):
        if len(x) < 2:
            return 0
        return sum(1 for i in range(1, len(x)) if x[i] < x[i-1]) / (len(x) - 1)

    result[f'{tf}_higher_highs_ratio'] = df['close'].rolling(window=seq_length).apply(
        lambda x: calc_higher_highs(x.values), raw=False
    ).fillna(0)

    result[f'{tf}_lower_lows_ratio'] = df['close'].rolling(window=seq_length).apply(
        lambda x: calc_lower_lows(x.values), raw=False
    ).fillna(0)

    # Position in range (0 = at min, 1 = at max)
    def calc_position_in_range(x):
        if len(x) < 1:
            return 0.5
        min_val = x.min()
        max_val = x.max()
        if max_val == min_val:
            return 0.5
        return (x.iloc[-1] - min_val) / (max_val - min_val)

    result[f'{tf}_position_in_range'] = df['close'].rolling(window=seq_length).apply(
        lambda x: calc_position_in_range(x), raw=False
    ).fillna(0.5)

    # Range expansion/contraction
    def calc_range_change(x):
        if len(x) < 2:
            return 0
        mid = len(x) // 2
        early_range = x[:mid].max() - x[:mid].min()
        late_range = x[mid:].max() - x[mid:].min()
        if early_range == 0:
            return 0
        return (late_range - early_range) / early_range

    result[f'{tf}_range_expansion'] = df['close'].rolling(window=seq_length).apply(
        lambda x: calc_range_change(x.values), raw=False
    ).fillna(0)

    return result


def calculate_volume_momentum(df: pd.DataFrame, timeframe: str, periods: List[int] = [3, 7, 14]) -> pd.DataFrame:
    """
    Volume momentum and trends

    Args:
        df: DataFrame with 'volume'
        timeframe: Timeframe identifier
        periods: List of periods

    Returns:
        DataFrame with 6 features (volume ratio + trend per period)
    """
    result = df.copy()
    tf = timeframe

    for period in periods:
        # Average volume ratio
        avg_volume = df['volume'].rolling(window=period).mean()
        result[f'{tf}_volume_ratio_{period}'] = (df['volume'] / avg_volume).fillna(1.0)

        # Volume trend (increasing/decreasing)
        def calc_volume_trend(x):
            if len(x) < 2:
                return 0
            mid = len(x) // 2
            early_avg = x[:mid].mean()
            late_avg = x[mid:].mean()
            if early_avg == 0:
                return 0
            return (late_avg - early_avg) / early_avg

        result[f'{tf}_volume_trend_{period}'] = df['volume'].rolling(window=period).apply(
            lambda x: calc_volume_trend(x.values), raw=False
        ).fillna(0)

    return result


def calculate_indicator_momentum(df: pd.DataFrame, timeframe: str, lookback: int = 3) -> pd.DataFrame:
    """
    Track momentum of key indicators (RSI, MACD, etc.)

    Args:
        df: DataFrame with indicator columns
        timeframe: Timeframe identifier
        lookback: Lookback period for momentum

    Returns:
        DataFrame with 6 indicator momentum features
    """
    result = df.copy()
    tf = timeframe

    # RSI momentum
    if f'{tf}_rsi_14' in df.columns:
        result[f'{tf}_rsi_momentum'] = df[f'{tf}_rsi_14'].diff(lookback).fillna(0)
    else:
        result[f'{tf}_rsi_momentum'] = 0

    # MACD momentum
    if f'{tf}_macd_line' in df.columns:
        result[f'{tf}_macd_momentum'] = df[f'{tf}_macd_line'].diff(lookback).fillna(0)
    else:
        result[f'{tf}_macd_momentum'] = 0

    # Stochastic momentum
    if f'{tf}_stoch_k' in df.columns:
        result[f'{tf}_stoch_momentum'] = df[f'{tf}_stoch_k'].diff(lookback).fillna(0)
    else:
        result[f'{tf}_stoch_momentum'] = 0

    # CMF momentum
    if f'{tf}_cmf_20' in df.columns:
        result[f'{tf}_cmf_momentum'] = df[f'{tf}_cmf_20'].diff(lookback).fillna(0)
    else:
        result[f'{tf}_cmf_momentum'] = 0

    # BB position momentum
    if f'{tf}_bb_percent' in df.columns:
        result[f'{tf}_bb_position_momentum'] = df[f'{tf}_bb_percent'].diff(lookback).fillna(0)
    else:
        result[f'{tf}_bb_position_momentum'] = 0

    # ADX momentum (trend strength change)
    if f'{tf}_adx_14' in df.columns:
        result[f'{tf}_adx_momentum'] = df[f'{tf}_adx_14'].diff(lookback).fillna(0)
    else:
        result[f'{tf}_adx_momentum'] = 0

    return result


def calculate_temporal_correlation(df: pd.DataFrame, timeframe: str, window: int = 14) -> pd.DataFrame:
    """
    Price-Volume correlation over time

    Args:
        df: DataFrame with 'close' and 'volume'
        timeframe: Timeframe identifier
        window: Correlation window

    Returns:
        DataFrame with 2 features (correlation + divergence)
    """
    result = df.copy()
    tf = timeframe

    # Correlation coefficient
    result[f'{tf}_price_volume_corr'] = df['close'].rolling(window=window).corr(df['volume']).fillna(0)

    # Divergence detection (price up, volume down or vice versa)
    price_trend = df['close'].pct_change(window)
    volume_trend = df['volume'].pct_change(window)

    result[f'{tf}_price_volume_divergence'] = (
        (np.sign(price_trend) != np.sign(volume_trend)).astype(int)
    )

    return result


def calculate_temporal_features(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Calculate all 49 temporal features for a given timeframe

    Args:
        df: DataFrame with base indicators already calculated
        timeframe: Timeframe identifier ('4h', '1d', '1w')

    Returns:
        DataFrame with original columns + 49 temporal features

    Feature breakdown:
        - Lag features: 10 (5 price + 5 volume)
        - Rolling statistics: 12 (4 windows × 3 metrics)
        - Momentum sequence: 9 (5 momentum + 4 acceleration)
        - Price sequence pattern: 4
        - Volume momentum: 6 (3 periods × 2)
        - Indicator momentum: 6
        - Temporal correlation: 2
        TOTAL: 49 features
    """
    result = df.copy()

    # 1. Lag features (10)
    result = calculate_lag_features(result, timeframe)

    # 2. Rolling statistics (12)
    result = calculate_rolling_statistics(result, timeframe)

    # 3. Momentum sequence (9)
    result = calculate_momentum_sequence(result, timeframe)

    # 4. Price sequence pattern (4)
    result = calculate_price_sequence_pattern(result, timeframe)

    # 5. Volume momentum (6)
    result = calculate_volume_momentum(result, timeframe)

    # 6. Indicator momentum (6) - requires base indicators
    result = calculate_indicator_momentum(result, timeframe)

    # 7. Temporal correlation (2)
    result = calculate_temporal_correlation(result, timeframe)

    return result


def calculate_multi_tf_temporal_features(
    df_4h: pd.DataFrame,
    df_1d: pd.DataFrame,
    df_1w: pd.DataFrame
) -> Dict[str, pd.DataFrame]:
    """
    Calculate temporal features for all 3 timeframes

    Args:
        df_4h: 4-hour dataframe (with base indicators)
        df_1d: 1-day dataframe (with base indicators)
        df_1w: 1-week dataframe (with base indicators)

    Returns:
        Dict with keys '4h', '1d', '1w' containing DataFrames with temporal features

    Feature count:
        - Per timeframe: 49 temporal features
        - Total: 147 temporal features (49 × 3 timeframes)
    """
    return {
        '4h': calculate_temporal_features(df_4h, '4h'),
        '1d': calculate_temporal_features(df_1d, '1d'),
        '1w': calculate_temporal_features(df_1w, '1w')
    }
