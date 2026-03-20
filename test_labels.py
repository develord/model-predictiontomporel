"""
Test Label Generation
=====================
Validate that label generation works correctly for all timeframes and cryptos
"""

import sys
import pandas as pd
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data.data_manager_multi_tf import get_dataframe
from features.labels import generate_labels, generate_multi_tf_labels, validate_labels, LOOKAHEAD_CONFIG


def test_labels():
    """Test label generation on BTC, ETH, SOL"""

    print("=" * 80)
    print("V10 LABEL GENERATION TEST")
    print("=" * 80)

    cryptos = ['btc', 'eth', 'sol']
    timeframes = ['4h', '1d', '1w']

    # Test each crypto
    for crypto in cryptos:
        print(f"\n{'=' * 80}")
        print(f"Testing {crypto.upper()} Label Generation")
        print('=' * 80)

        # Load data for all timeframes
        dfs = {}
        for tf in timeframes:
            df = get_dataframe(crypto, tf)
            if df is None:
                print(f"  X Failed to load {crypto} {tf} data")
                continue
            dfs[tf] = df
            print(f"  + Loaded {tf}: {len(df)} rows")

        if len(dfs) != 3:
            print(f"  X Skipping {crypto} - missing data")
            continue

        # Generate labels for all timeframes
        print(f"\n  Generating labels...")
        labeled_dfs, stats = generate_multi_tf_labels(
            dfs['4h'],
            dfs['1d'],
            dfs['1w'],
            crypto=crypto
        )

        # Display results for each timeframe
        for tf in timeframes:
            print(f"\n  [{tf.upper()}] Label Statistics:")

            tf_stats = stats[tf]
            lookahead = LOOKAHEAD_CONFIG[tf]

            print(f"    Lookahead: {lookahead['candles']} candles ({lookahead['description']})")

            # Classification distribution
            class_stats = tf_stats['classification']
            counts = class_stats['counts']
            pcts = class_stats['percentages']

            print(f"    Classification Distribution:")
            print(f"      BUY:  {counts['BUY']:5d} ({pcts['BUY']:5.1f}%)")
            print(f"      HOLD: {counts['HOLD']:5d} ({pcts['HOLD']:5.1f}%)")
            print(f"      SELL: {counts['SELL']:5d} ({pcts['SELL']:5.1f}%)")
            print(f"      Total: {counts['total']} valid labels")

            # Regression statistics
            reg_stats = tf_stats['regression']
            print(f"    Regression Statistics (price_target_pct):")
            print(f"      Mean:   {reg_stats['mean']:+6.2f}%")
            print(f"      Median: {reg_stats['median']:+6.2f}%")
            print(f"      Std:    {reg_stats['std']:6.2f}%")
            print(f"      Range:  {reg_stats['min']:+6.2f}% to {reg_stats['max']:+6.2f}%")
            print(f"      Q25-Q75: {reg_stats['q25']:+6.2f}% to {reg_stats['q75']:+6.2f}%")

            # Validation
            validations = validate_labels(labeled_dfs[tf], tf)
            print(f"    Validation:")
            print(f"      All checks passed: {validations['all_valid']}")
            if not validations['all_valid']:
                print(f"      Failed checks:")
                for check, passed in validations.items():
                    if not passed and check != 'all_valid':
                        print(f"        - {check}")

            # NaN info
            print(f"    NaN rows: {tf_stats['nan_count']} (last {lookahead['candles']} candles + warmup)")

        # Sample labels
        print(f"\n  Sample Labels (last 10 valid rows from 1d):")
        df_1d = labeled_dfs['1d']
        valid_rows = df_1d[df_1d['label_class'].notna()].tail(10)

        print(f"  {'Index':<8} {'Close':>10} {'Future':>10} {'Target%':>8} {'Label':>6}")
        print(f"  {'-'*55}")

        for idx, row in valid_rows.iterrows():
            close = row['close']
            future = row['future_price']
            target = row['price_target_pct']
            label = row['label_class']

            print(f"  {idx:<8} {close:>10.2f} {future:>10.2f} {target:>+7.2f}% {label:>6}")

    # Final summary
    print("\n" + "=" * 80)
    print("LABEL GENERATION SUMMARY")
    print("=" * 80)
    print()
    print("Expected behavior:")
    print("  - BUY labels: price_target_pct > +3% (or crypto-specific threshold)")
    print("  - SELL labels: price_target_pct < -3%")
    print("  - HOLD labels: price_target_pct between -3% and +3%")
    print()
    print("Lookahead periods:")
    print("  - 4h: 42 candles = 7 days")
    print("  - 1d:  7 candles = 7 days (V9 standard)")
    print("  - 1w:  1 candle  = 1 week")
    print()
    print("Regression labels:")
    print("  - Used for DYNAMIC TP/SL calculation")
    print("  - TP = price_target_pct × 0.75")
    print("  - SL = price_target_pct × 0.35")
    print()
    print("=" * 80)
    print("TEST COMPLETE!")
    print("=" * 80)

    return True


if __name__ == '__main__':
    success = test_labels()
    sys.exit(0 if success else 1)
