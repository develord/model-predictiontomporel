"""
V10 Multi-Timeframe Feature Pipeline
====================================
Merge features from 4h, 1d, and 1w timeframes into a single dataset

Key Challenges:
1. Temporal alignment - 1d is the PRIMARY timeframe
   - 4h data needs to be resampled to 1d
   - 1w data needs to be forward-filled to 1d
2. Feature merging - combine 237-348 features from 3 timeframes
3. Label selection - use 1d labels as primary (7-day lookahead)

Architecture:
- Primary timeframe: 1d (this drives the timestamps)
- 4h features: Use the LAST 4h candle of each day
- 1w features: Forward-fill from weekly to daily
- Labels: From 1d timeframe (7 days lookahead)

Result:
- BTC: 237 features (79 × 3 timeframes)
- ETH/SOL: 348 features (116 × 3 timeframes)
- All aligned to 1d timestamps
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.data_manager_multi_tf import get_dataframe
from features.base_indicators import calculate_base_indicators
from features.temporal_features import calculate_temporal_features
from features.btc_influence import calculate_btc_influence_features
from features.labels import generate_labels


def align_4h_to_1d(df_4h: pd.DataFrame) -> pd.DataFrame:
    """
    Align 4h data to 1d timestamps by taking the LAST 4h candle of each day

    Args:
        df_4h: 4-hour DataFrame with features

    Returns:
        DataFrame resampled to daily with last values
    """
    # Resample to daily, taking last value of each day
    # This represents the state at the end of each trading day
    df_1d_aligned = df_4h.resample('1D').last()

    # Forward fill any missing days (weekends, holidays)
    df_1d_aligned = df_1d_aligned.fillna(method='ffill')

    return df_1d_aligned


def align_1w_to_1d(df_1w: pd.DataFrame) -> pd.DataFrame:
    """
    Align 1w data to 1d timestamps by forward-filling weekly values

    Args:
        df_1w: 1-week DataFrame with features

    Returns:
        DataFrame resampled to daily with forward-filled values
    """
    # Resample to daily with forward fill
    # This means each week's values are constant for 7 days
    df_1d_aligned = df_1w.resample('1D').ffill()

    return df_1d_aligned


def merge_multi_tf_features(
    df_4h: pd.DataFrame,
    df_1d: pd.DataFrame,
    df_1w: pd.DataFrame,
    crypto: str = 'btc'
) -> pd.DataFrame:
    """
    Merge features from all 3 timeframes into single dataset

    Args:
        df_4h: 4-hour DataFrame with features (already calculated)
        df_1d: 1-day DataFrame with features (already calculated)
        df_1w: 1-week DataFrame with features (already calculated)
        crypto: Crypto identifier for logging

    Returns:
        Single DataFrame with all features aligned to 1d timeframe
    """
    print(f"\n  Merging multi-TF features for {crypto.upper()}...")

    # Step 1: Align 4h and 1w to 1d timestamps
    print(f"    Aligning 4h to 1d (resample last)...")
    df_4h_aligned = align_4h_to_1d(df_4h)
    print(f"      4h: {len(df_4h)} rows -> {len(df_4h_aligned)} rows")

    print(f"    Aligning 1w to 1d (forward fill)...")
    df_1w_aligned = align_1w_to_1d(df_1w)
    print(f"      1w: {len(df_1w)} rows -> {len(df_1w_aligned)} rows")

    # Step 2: Start with 1d as base (this has OHLCV + labels)
    result = df_1d.copy()

    # Step 3: Merge 4h features (exclude OHLCV, keep only feature columns)
    exclude_cols = ['open', 'high', 'low', 'close', 'volume',
                    'label_class', 'label_numeric', 'price_target_pct', 'future_price']

    # Get 4h feature columns
    feature_cols_4h = [col for col in df_4h_aligned.columns if col not in exclude_cols]

    # Merge 4h features by index (date)
    for col in feature_cols_4h:
        if col in df_4h_aligned.columns:
            result[col] = df_4h_aligned[col]

    print(f"    Added {len(feature_cols_4h)} features from 4h")

    # Step 4: Merge 1w features
    feature_cols_1w = [col for col in df_1w_aligned.columns if col not in exclude_cols]

    for col in feature_cols_1w:
        if col in df_1w_aligned.columns:
            result[col] = df_1w_aligned[col]

    print(f"    Added {len(feature_cols_1w)} features from 1w")

    # Step 5: Handle any remaining NaN (forward fill then back fill)
    nan_before = result.isnull().sum().sum()
    if nan_before > 0:
        result = result.fillna(method='ffill').fillna(method='bfill').fillna(0)
        print(f"    Filled {nan_before} NaN values")

    total_features = len(feature_cols_4h) + len([c for c in df_1d.columns if c not in exclude_cols]) + len(feature_cols_1w)
    print(f"    Total features: {total_features}")
    print(f"    Final shape: {result.shape}")

    return result


def build_multi_tf_dataset(
    crypto: str,
    include_btc_data: Dict[str, pd.DataFrame] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Build complete multi-timeframe dataset for a crypto

    Args:
        crypto: 'btc', 'eth', or 'sol'
        include_btc_data: Dict of BTC DataFrames for altcoin influence features

    Returns:
        Tuple of (merged DataFrame, statistics dict)
    """
    print(f"\n{'='*80}")
    print(f"Building Multi-TF Dataset: {crypto.upper()}")
    print('='*80)

    timeframes = ['4h', '1d', '1w']

    # Step 1: Load raw data for all timeframes
    print(f"\n[1/4] Loading Data")
    dfs_raw = {}
    for tf in timeframes:
        df = get_dataframe(crypto, tf)
        if df is None:
            raise ValueError(f"Failed to load {crypto} {tf} data")
        dfs_raw[tf] = df
        print(f"  + {tf}: {len(df)} rows")

    # Step 2: Calculate features for each timeframe
    print(f"\n[2/4] Calculating Features")
    dfs_with_features = {}

    for tf in timeframes:
        print(f"\n  [{tf.upper()}]")
        df = dfs_raw[tf].copy()

        # Base indicators
        df = calculate_base_indicators(df, tf)

        # Temporal features
        df = calculate_temporal_features(df, tf)

        # BTC influence (only for altcoins)
        if crypto != 'btc' and include_btc_data is not None:
            symbol = f"{crypto.upper()}USDT"
            df = calculate_btc_influence_features(
                df,
                include_btc_data[tf],
                tf,
                symbol
            )

        dfs_with_features[tf] = df
        print(f"    Total columns: {len(df.columns)}")

    # Step 3: Generate labels (on 1d timeframe)
    print(f"\n[3/4] Generating Labels (1d timeframe)")
    dfs_with_features['1d'], label_stats = generate_labels(
        dfs_with_features['1d'],
        '1d',
        crypto
    )

    print(f"  Label distribution:")
    print(f"    BUY:  {label_stats['classification']['counts']['BUY']:4d} ({label_stats['classification']['percentages']['BUY']:5.1f}%)")
    print(f"    HOLD: {label_stats['classification']['counts']['HOLD']:4d} ({label_stats['classification']['percentages']['HOLD']:5.1f}%)")
    print(f"    SELL: {label_stats['classification']['counts']['SELL']:4d} ({label_stats['classification']['percentages']['SELL']:5.1f}%)")

    # Step 4: Merge all timeframes
    print(f"\n[4/4] Merging Timeframes")
    merged_df = merge_multi_tf_features(
        dfs_with_features['4h'],
        dfs_with_features['1d'],
        dfs_with_features['1w'],
        crypto
    )

    # Final statistics
    stats = {
        'crypto': crypto,
        'rows': len(merged_df),
        'total_columns': len(merged_df.columns),
        'feature_columns': len(merged_df.columns) - 9,  # Exclude OHLCV + 4 label columns
        'label_stats': label_stats,
        'timeframes': {
            '4h': len(dfs_with_features['4h'].columns),
            '1d': len(dfs_with_features['1d'].columns),
            '1w': len(dfs_with_features['1w'].columns)
        }
    }

    return merged_df, stats


def build_all_cryptos() -> Dict[str, Tuple[pd.DataFrame, Dict]]:
    """
    Build multi-TF datasets for all cryptos (BTC, ETH, SOL)

    Returns:
        Dict mapping crypto -> (DataFrame, stats)
    """
    print("="*80)
    print("V10 MULTI-TIMEFRAME PIPELINE")
    print("="*80)

    cryptos = ['btc', 'eth', 'sol']
    timeframes = ['4h', '1d', '1w']

    # Step 1: Load BTC data for influence features
    print("\n[BTC DATA] Loading for altcoin influence features...")
    btc_data = {}
    for tf in timeframes:
        df = get_dataframe('btc', tf)
        if df is None:
            raise ValueError(f"Failed to load BTC {tf} data")
        btc_data[tf] = df
        print(f"  + BTC {tf}: {len(df)} rows")

    # Step 2: Build datasets for each crypto
    results = {}
    for crypto in cryptos:
        # BTC doesn't need BTC influence features
        btc_input = None if crypto == 'btc' else btc_data

        df, stats = build_multi_tf_dataset(crypto, btc_input)
        results[crypto] = (df, stats)

    # Summary
    print(f"\n{'='*80}")
    print("PIPELINE SUMMARY")
    print('='*80)

    for crypto in cryptos:
        df, stats = results[crypto]
        print(f"\n{crypto.upper()}:")
        print(f"  Rows: {stats['rows']}")
        print(f"  Features: {stats['feature_columns']}")
        print(f"  Total columns: {stats['total_columns']}")
        print(f"  Valid labels: {stats['label_stats']['valid_count']}")

    return results


if __name__ == '__main__':
    # Test the pipeline
    results = build_all_cryptos()

    # Save to cache for training
    cache_dir = Path(__file__).parent.parent / 'data' / 'cache'
    cache_dir.mkdir(parents=True, exist_ok=True)

    for crypto, (df, stats) in results.items():
        output_file = cache_dir / f'{crypto}_multi_tf_merged.csv'
        df.to_csv(output_file)
        print(f"\nSaved {crypto.upper()} to {output_file}")

    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)
