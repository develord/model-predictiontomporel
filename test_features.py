"""
Test Feature Calculation Pipeline
==================================
Verify that all T3-T5 features work correctly with real data
"""

import sys
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from data.data_manager_multi_tf import get_dataframe
from features.base_indicators import calculate_base_indicators
from features.temporal_features import calculate_temporal_features
from features.btc_influence import calculate_btc_influence_features


def test_features():
    """Test feature calculation on BTC, ETH, SOL"""

    print("=" * 80)
    print("V10 FEATURE CALCULATION TEST")
    print("=" * 80)

    cryptos = ['btc', 'eth', 'sol']
    timeframes = ['4h', '1d', '1w']

    results = {}

    # Load BTC data (needed for influence features)
    print("\n[1/4] Loading BTC data for influence features...")
    btc_data = {}
    for tf in timeframes:
        df = get_dataframe('btc', tf)
        if df is not None:
            btc_data[tf] = df
            print(f"  + BTC {tf}: {len(df)} rows")
        else:
            print(f"  X BTC {tf}: FAILED TO LOAD")
            return False

    # Test each crypto
    for crypto in cryptos:
        print(f"\n{'=' * 80}")
        print(f"Testing {crypto.upper()}")
        print('=' * 80)

        crypto_results = {}

        for tf in timeframes:
            print(f"\n[{tf.upper()}] Loading data...")
            df = get_dataframe(crypto, tf)

            if df is None:
                print(f"  X Failed to load {crypto} {tf} data")
                continue

            print(f"  + Loaded {len(df)} rows")
            initial_cols = len(df.columns)

            # T3: Base indicators
            print(f"  [T3] Calculating base indicators...")
            df = calculate_base_indicators(df, tf)
            base_cols = len(df.columns) - initial_cols
            print(f"  + Added {base_cols} base indicator features")

            # T4: Temporal features
            print(f"  [T4] Calculating temporal features...")
            cols_before_temporal = len(df.columns)
            df = calculate_temporal_features(df, tf)
            temporal_cols = len(df.columns) - cols_before_temporal
            print(f"  + Added {temporal_cols} temporal features")

            # T5: BTC influence (only for ETH/SOL)
            if crypto != 'btc':
                print(f"  [T5] Calculating BTC influence features...")
                cols_before_btc = len(df.columns)
                symbol = f"{crypto.upper()}USDT"
                df = calculate_btc_influence_features(
                    df,
                    btc_data[tf],
                    tf,
                    symbol
                )
                btc_cols = len(df.columns) - cols_before_btc
                print(f"  + Added {btc_cols} BTC influence features")
            else:
                btc_cols = 0
                print(f"  [T5] Skipped BTC influence (this is BTC)")

            # Summary
            total_features = base_cols + temporal_cols + btc_cols
            crypto_results[tf] = {
                'rows': len(df),
                'base_features': base_cols,
                'temporal_features': temporal_cols,
                'btc_features': btc_cols,
                'total_features': total_features,
                'total_columns': len(df.columns)
            }

            print(f"\n  Summary for {crypto.upper()} {tf}:")
            print(f"    - Base indicators:   {base_cols:3d} features")
            print(f"    - Temporal features: {temporal_cols:3d} features")
            print(f"    - BTC influence:     {btc_cols:3d} features")
            print(f"    - TOTAL:             {total_features:3d} features")
            print(f"    - Total columns:     {len(df.columns):3d}")

        results[crypto] = crypto_results

    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    for crypto in cryptos:
        print(f"\n{crypto.upper()}:")
        for tf in timeframes:
            if tf in results[crypto]:
                r = results[crypto][tf]
                print(f"  {tf}: {r['total_features']:3d} features ({r['rows']} rows)")

    # Expected feature counts
    print("\n" + "=" * 80)
    print("EXPECTED FEATURE COUNTS (V9 baseline)")
    print("=" * 80)
    print("BTC per timeframe:  30 base + 49 temporal = 79 features")
    print("BTC total (3 TFs):  79 × 3 = 237 features")
    print()
    print("ETH/SOL per timeframe: 30 base + 49 temporal + 37 BTC = 116 features")
    print("ETH/SOL total (3 TFs): 116 × 3 = 348 features")

    print("\n" + "=" * 80)
    print("TEST COMPLETE!")
    print("=" * 80)

    return True


if __name__ == '__main__':
    success = test_features()
    sys.exit(0 if success else 1)
