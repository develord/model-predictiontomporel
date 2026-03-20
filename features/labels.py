"""
V10 Label Generation - Dual Model System
=========================================
Generate labels for both classification and regression models

Classification Labels:
- BUY: Price increases significantly (>threshold)
- SELL: Price decreases significantly (<-threshold)
- HOLD: Price movement within threshold range

Regression Labels:
- price_target_pct: Exact percentage move (e.g., +8.5%, -3.2%)
- Used for dynamic TP/SL calculation

This is the KEY INNOVATION of V10:
Instead of fixed TP/SL, we predict the magnitude of the move
and adapt our targets accordingly.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
import json
from pathlib import Path


# Lookahead configuration by timeframe
LOOKAHEAD_CONFIG = {
    '4h': {
        'candles': 42,  # 7 days (6 candles/day × 7)
        'description': '7 days ahead'
    },
    '1d': {
        'candles': 7,   # 7 days (V9 standard)
        'description': '7 days ahead'
    },
    '1w': {
        'candles': 1,   # 1 week
        'description': '1 week ahead'
    }
}


def load_crypto_config(crypto: str) -> Dict:
    """
    Load crypto-specific configuration

    Args:
        crypto: 'btc', 'eth', or 'sol'

    Returns:
        Dict with crypto configuration
    """
    config_path = Path(__file__).parent.parent / 'config' / 'cryptos.json'

    with open(config_path, 'r') as f:
        config = json.load(f)

    crypto_key = crypto.upper()
    if crypto_key not in config:
        raise ValueError(f"Crypto {crypto} not found in config")

    return config[crypto_key]


def calculate_price_target(df: pd.DataFrame, lookahead_candles: int) -> pd.Series:
    """
    Calculate future price for regression target

    Args:
        df: DataFrame with 'close' column
        lookahead_candles: How many periods to look ahead

    Returns:
        Series of future prices (with NaN for last N rows)
    """
    return df['close'].shift(-lookahead_candles)


def generate_regression_labels(
    df: pd.DataFrame,
    lookahead_candles: int,
    clip_percentile: float = 99.0
) -> pd.DataFrame:
    """
    Generate regression labels (price_target_pct)

    Args:
        df: DataFrame with 'close' column
        lookahead_candles: How many periods to look ahead
        clip_percentile: Percentile to clip outliers (default 99%)

    Returns:
        DataFrame with 'price_target_pct' column added
    """
    result = df.copy()

    # Calculate future price
    future_price = calculate_price_target(df, lookahead_candles)

    # Calculate percentage change
    price_target_pct = ((future_price - df['close']) / df['close']) * 100

    # Clip extreme outliers to prevent overfitting on rare events
    if clip_percentile < 100:
        upper_clip = np.percentile(price_target_pct.dropna(), clip_percentile)
        lower_clip = np.percentile(price_target_pct.dropna(), 100 - clip_percentile)
        price_target_pct = price_target_pct.clip(lower=lower_clip, upper=upper_clip)

    result['price_target_pct'] = price_target_pct
    result['future_price'] = future_price

    return result


def generate_classification_labels(
    df: pd.DataFrame,
    buy_threshold: float = 3.0,
    sell_threshold: float = -3.0
) -> pd.DataFrame:
    """
    Generate classification labels (BUY/SELL/HOLD)

    Args:
        df: DataFrame with 'price_target_pct' column
        buy_threshold: Minimum % move to label as BUY (default 3%)
        sell_threshold: Maximum % move to label as SELL (default -3%)

    Returns:
        DataFrame with 'label_class' and 'label_numeric' columns added
    """
    result = df.copy()

    if 'price_target_pct' not in result.columns:
        raise ValueError("Must run generate_regression_labels first")

    # Initialize all as HOLD (0)
    result['label_class'] = 'HOLD'
    result['label_numeric'] = 0

    # BUY if price increases above threshold
    buy_mask = result['price_target_pct'] > buy_threshold
    result.loc[buy_mask, 'label_class'] = 'BUY'
    result.loc[buy_mask, 'label_numeric'] = 1

    # SELL if price decreases below threshold
    sell_mask = result['price_target_pct'] < sell_threshold
    result.loc[sell_mask, 'label_class'] = 'SELL'
    result.loc[sell_mask, 'label_numeric'] = -1

    return result


def calculate_label_statistics(df: pd.DataFrame, timeframe: str) -> Dict:
    """
    Calculate statistics about label distribution

    Args:
        df: DataFrame with labels
        timeframe: Timeframe identifier

    Returns:
        Dict with label statistics
    """
    valid_labels = df[df['label_class'].notna()]

    # Classification distribution
    class_counts = valid_labels['label_class'].value_counts()
    total = len(valid_labels)

    class_dist = {
        'BUY': int(class_counts.get('BUY', 0)),
        'HOLD': int(class_counts.get('HOLD', 0)),
        'SELL': int(class_counts.get('SELL', 0)),
        'total': total
    }

    class_pct = {
        'BUY': class_dist['BUY'] / total * 100 if total > 0 else 0,
        'HOLD': class_dist['HOLD'] / total * 100 if total > 0 else 0,
        'SELL': class_dist['SELL'] / total * 100 if total > 0 else 0
    }

    # Regression statistics
    price_targets = valid_labels['price_target_pct'].dropna()

    reg_stats = {
        'mean': float(price_targets.mean()),
        'std': float(price_targets.std()),
        'min': float(price_targets.min()),
        'max': float(price_targets.max()),
        'median': float(price_targets.median()),
        'q25': float(price_targets.quantile(0.25)),
        'q75': float(price_targets.quantile(0.75))
    }

    return {
        'timeframe': timeframe,
        'classification': {
            'counts': class_dist,
            'percentages': class_pct
        },
        'regression': reg_stats,
        'nan_count': int(df['label_class'].isna().sum()),
        'valid_count': total
    }


def generate_labels(
    df: pd.DataFrame,
    timeframe: str,
    crypto: str = 'btc',
    buy_threshold: float = None,
    sell_threshold: float = None,
    clip_percentile: float = 99.0
) -> Tuple[pd.DataFrame, Dict]:
    """
    Generate both classification and regression labels for a timeframe

    Args:
        df: DataFrame with OHLCV data
        timeframe: '4h', '1d', or '1w'
        crypto: 'btc', 'eth', or 'sol'
        buy_threshold: Custom BUY threshold (default from config)
        sell_threshold: Custom SELL threshold (default from config)
        clip_percentile: Percentile to clip outliers

    Returns:
        Tuple of (DataFrame with labels, statistics dict)
    """
    if timeframe not in LOOKAHEAD_CONFIG:
        raise ValueError(f"Invalid timeframe: {timeframe}. Must be 4h, 1d, or 1w")

    # Get lookahead configuration
    lookahead_candles = LOOKAHEAD_CONFIG[timeframe]['candles']

    # Load crypto config for thresholds if not provided
    if buy_threshold is None or sell_threshold is None:
        crypto_config = load_crypto_config(crypto)
        # Use min_magnitude_pct as threshold
        magnitude = crypto_config.get('min_magnitude_pct', 3.0)
        buy_threshold = buy_threshold or magnitude
        sell_threshold = sell_threshold or -magnitude

    # Step 1: Generate regression labels (price_target_pct)
    result = generate_regression_labels(df, lookahead_candles, clip_percentile)

    # Step 2: Generate classification labels (BUY/SELL/HOLD)
    result = generate_classification_labels(result, buy_threshold, sell_threshold)

    # Step 3: Calculate statistics
    stats = calculate_label_statistics(result, timeframe)

    return result, stats


def generate_multi_tf_labels(
    df_4h: pd.DataFrame,
    df_1d: pd.DataFrame,
    df_1w: pd.DataFrame,
    crypto: str = 'btc'
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict]]:
    """
    Generate labels for all 3 timeframes

    Args:
        df_4h: 4-hour DataFrame
        df_1d: 1-day DataFrame
        df_1w: 1-week DataFrame
        crypto: 'btc', 'eth', or 'sol'

    Returns:
        Tuple of (dict of DataFrames with labels, dict of statistics)
    """
    results = {}
    stats = {}

    for tf, df in [('4h', df_4h), ('1d', df_1d), ('1w', df_1w)]:
        df_with_labels, tf_stats = generate_labels(df, tf, crypto)
        results[tf] = df_with_labels
        stats[tf] = tf_stats

    return results, stats


def validate_labels(df: pd.DataFrame, timeframe: str) -> Dict[str, bool]:
    """
    Validate that labels are properly generated

    Args:
        df: DataFrame with labels
        timeframe: Timeframe identifier

    Returns:
        Dict with validation results
    """
    validations = {}

    # Check required columns exist
    validations['has_price_target_pct'] = 'price_target_pct' in df.columns
    validations['has_label_class'] = 'label_class' in df.columns
    validations['has_label_numeric'] = 'label_numeric' in df.columns
    validations['has_future_price'] = 'future_price' in df.columns

    # Check no infinite values
    if validations['has_price_target_pct']:
        validations['no_inf_regression'] = not np.isinf(df['price_target_pct'].dropna()).any()

    # Check label classes are valid
    if validations['has_label_class']:
        valid_classes = {'BUY', 'SELL', 'HOLD'}
        actual_classes = set(df['label_class'].dropna().unique())
        validations['valid_classes'] = actual_classes.issubset(valid_classes)

    # Check label numerics are valid
    if validations['has_label_numeric']:
        valid_numerics = {-1, 0, 1}
        actual_numerics = set(df['label_numeric'].dropna().unique())
        validations['valid_numerics'] = actual_numerics.issubset(valid_numerics)

    # Check NaN count is reasonable (only last N rows should be NaN)
    lookahead = LOOKAHEAD_CONFIG[timeframe]['candles']
    nan_count = df['label_class'].isna().sum()
    validations['reasonable_nan_count'] = nan_count <= lookahead + 10  # Allow some buffer

    validations['all_valid'] = all(validations.values())

    return validations
