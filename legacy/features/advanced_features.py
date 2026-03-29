"""
Advanced Features - Market Structure & Volume Analysis
======================================================
Additional features to improve model performance:

1. Market Structure (8 features):
   - Higher Highs / Lower Lows detection
   - Support/Resistance levels
   - Trend strength
   - Price structure score

2. Advanced Volume (7 features):
   - Volume momentum
   - Abnormal volume detection
   - Volume trend
   - Price-Volume divergence

3. Fibonacci Levels (4 features):
   - Distance from key Fib levels
   - Fib retracement zones

Total: 19 new features
"""

import numpy as np
import pandas as pd
from typing import Dict


def detect_higher_highs_lower_lows(df: pd.DataFrame, window: int = 20) -> Dict[str, pd.Series]:
    """
    Detect Higher Highs (HH) and Lower Lows (LL) pattern
    """
    high = df['high']
    low = df['low']

    prev_high = high.rolling(window=window).max().shift(1)
    prev_low = low.rolling(window=window).min().shift(1)

    is_hh = (high > prev_high).astype(int)
    is_ll = (low < prev_low).astype(int)

    hh_count = is_hh.rolling(window=window).sum()
    ll_count = is_ll.rolling(window=window).sum()

    return {
        'is_hh': is_hh,
        'is_ll': is_ll,
        'hh_count': hh_count,
        'll_count': ll_count
    }


def calculate_support_resistance(df: pd.DataFrame, window: int = 50) -> Dict[str, pd.Series]:
    """
    Calculate dynamic support and resistance levels
    """
    close = df['close']
    high = df['high']
    low = df['low']

    resistance = high.rolling(window=window).max()
    support = low.rolling(window=window).min()

    dist_to_resistance = ((resistance - close) / close * 100)
    dist_to_support = ((close - support) / close * 100)

    range_position = (close - support) / (resistance - support + 1e-10)

    return {
        'dist_to_resistance_pct': dist_to_resistance,
        'dist_to_support_pct': dist_to_support,
        'range_position': range_position
    }


def calculate_trend_strength(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Calculate trend strength based on price momentum and consistency
    """
    close = df['close']

    price_change = close.pct_change(window)

    returns = close.pct_change()
    up_days = (returns > 0).rolling(window=window).sum() / window
    down_days = (returns < 0).rolling(window=window).sum() / window

    trend_strength = price_change * (up_days - down_days)

    return trend_strength


def calculate_volume_momentum(df: pd.DataFrame, fast: int = 5, slow: int = 20) -> Dict[str, pd.Series]:
    """
    Calculate volume momentum and trends
    """
    volume = df['volume']

    vol_fast = volume.rolling(window=fast).mean()
    vol_slow = volume.rolling(window=slow).mean()

    vol_momentum = (vol_fast / vol_slow - 1) * 100
    vol_trend = (volume / vol_slow - 1) * 100
    vol_acceleration = vol_momentum.diff()

    return {
        'vol_momentum': vol_momentum,
        'vol_trend': vol_trend,
        'vol_acceleration': vol_acceleration
    }


def detect_abnormal_volume(df: pd.DataFrame, window: int = 20, threshold: float = 2.0) -> Dict[str, pd.Series]:
    """
    Detect abnormal volume spikes
    """
    volume = df['volume']

    vol_mean = volume.rolling(window=window).mean()
    vol_std = volume.rolling(window=window).std()
    vol_zscore = (volume - vol_mean) / (vol_std + 1e-10)

    is_vol_spike = (vol_zscore > threshold).astype(int)

    return {
        'vol_zscore': vol_zscore,
        'is_vol_spike': is_vol_spike
    }


def calculate_price_volume_divergence(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Detect price-volume divergence
    """
    close = df['close']
    volume = df['volume']

    price_trend = close.pct_change(window)
    vol_trend = volume.pct_change(window)

    price_direction = np.sign(price_trend)
    vol_direction = np.sign(vol_trend)

    divergence = price_direction * vol_direction * -1

    return divergence


def calculate_fibonacci_levels(df: pd.DataFrame, window: int = 50) -> Dict[str, pd.Series]:
    """
    Calculate distance to Fibonacci retracement levels
    """
    close = df['close']
    high = df['high']
    low = df['low']

    swing_high = high.rolling(window=window).max()
    swing_low = low.rolling(window=window).min()

    fib_range = swing_high - swing_low

    fib_0_236 = swing_high - fib_range * 0.236
    fib_0_382 = swing_high - fib_range * 0.382
    fib_0_5 = swing_high - fib_range * 0.5
    fib_0_618 = swing_high - fib_range * 0.618

    dist_to_236 = abs(close - fib_0_236) / close * 100
    dist_to_382 = abs(close - fib_0_382) / close * 100
    dist_to_5 = abs(close - fib_0_5) / close * 100
    dist_to_618 = abs(close - fib_0_618) / close * 100

    all_dists = pd.concat([dist_to_236, dist_to_382, dist_to_5, dist_to_618], axis=1)
    closest_fib_dist = all_dists.min(axis=1)

    return {
        'closest_fib_dist_pct': closest_fib_dist,
        'fib_236_dist': dist_to_236,
        'fib_382_dist': dist_to_382,
        'fib_618_dist': dist_to_618
    }


def calculate_pivot_points(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Calculate pivot points (daily)
    """
    high = df['high'].shift(1)
    low = df['low'].shift(1)
    close = df['close'].shift(1)

    pivot = (high + low + close) / 3

    r1 = 2 * pivot - low
    s1 = 2 * pivot - high

    current_close = df['close']
    dist_to_pivot = ((current_close - pivot) / current_close * 100)
    dist_to_r1 = ((r1 - current_close) / current_close * 100)
    dist_to_s1 = ((current_close - s1) / current_close * 100)

    return {
        'dist_to_pivot_pct': dist_to_pivot,
        'dist_to_r1_pct': dist_to_r1,
        'dist_to_s1_pct': dist_to_s1
    }


def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all advanced features to dataframe
    """
    df = df.copy()

    # 1. Market Structure
    hh_ll = detect_higher_highs_lower_lows(df)
    for key, series in hh_ll.items():
        df[f'adv_{key}'] = series

    sr = calculate_support_resistance(df)
    for key, series in sr.items():
        df[f'adv_{key}'] = series

    df['adv_trend_strength'] = calculate_trend_strength(df)

    # 2. Advanced Volume
    vol_mom = calculate_volume_momentum(df)
    for key, series in vol_mom.items():
        df[f'adv_{key}'] = series

    vol_anom = detect_abnormal_volume(df)
    for key, series in vol_anom.items():
        df[f'adv_{key}'] = series

    df['adv_pv_divergence'] = calculate_price_volume_divergence(df)

    # 3. Fibonacci levels
    fib = calculate_fibonacci_levels(df)
    for key, series in fib.items():
        df[f'adv_{key}'] = series

    # 4. Pivot points
    pivots = calculate_pivot_points(df)
    for key, series in pivots.items():
        df[f'adv_{key}'] = series

    return df
