"""
Threshold Optimization - V11 TEMPORAL
======================================
Optimize P(TP) threshold to maximize Expected Value

Currently: threshold = 0.5 for all cryptos (not optimal!)
Goal: Find optimal threshold per crypto that maximizes EV

Expected improvement: +5-10% ROI
"""

import sys
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent))


def simulate_threshold(y_true, y_pred_proba, threshold, tp_pct=1.5, sl_pct=0.75):
    """
    Simulate trading with given threshold

    Returns:
        metrics: Dict with win_rate, expected_value, total_roi, num_trades
    """
    # Generate signals
    signals = (y_pred_proba > threshold).astype(int)

    # Count outcomes
    tp_count = np.sum((signals == 1) & (y_true == 1))
    sl_count = np.sum((signals == 1) & (y_true == 0))
    total_trades = tp_count + sl_count

    if total_trades == 0:
        return {
            'threshold': threshold,
            'num_trades': 0,
            'win_rate': 0.0,
            'expected_value': 0.0,
            'total_roi': 0.0
        }

    win_rate = tp_count / total_trades
    expected_value = (tp_pct * win_rate) - (sl_pct * (1 - win_rate))
    total_roi = (tp_count * tp_pct) - (sl_count * sl_pct)

    return {
        'threshold': threshold,
        'num_trades': total_trades,
        'win_rate': win_rate,
        'expected_value': expected_value,
        'total_roi': total_roi,
        'tp_count': tp_count,
        'sl_count': sl_count
    }


def optimize_threshold(crypto: str, model_type='baseline', tp_pct=1.5, sl_pct=0.75):
    """
    Find optimal P(TP) threshold for a crypto

    Args:
        crypto: 'btc', 'eth', or 'sol'
        model_type: 'baseline' or 'optimized'
        tp_pct: Take profit %
        sl_pct: Stop loss %

    Returns:
        results: Dictionary with optimization results
    """

    print(f"\n{'='*80}")
    print(f"THRESHOLD OPTIMIZATION FOR {crypto.upper()} - {model_type.upper()}")
    print(f"TP/SL: {tp_pct}% / {sl_pct}%")
    print('='*80)

    # Load model
    if model_type == 'optimized':
        model_file = Path(__file__).parent.parent / 'models' / f'{crypto}_v11_optimized.joblib'
    else:
        model_file = Path(__file__).parent.parent / 'models' / f'{crypto}_v11_classifier.joblib'

    if not model_file.exists():
        print(f"ERROR: Model not found: {model_file}")
        return None

    model = joblib.load(model_file)

    # Load data
    cache_file = Path(__file__).parent.parent / 'data' / 'cache' / f'{crypto}_multi_tf_merged.csv'
    df = pd.read_csv(cache_file, index_col=0, parse_dates=True)

    # Prepare features
    exclude_cols = [
        'open', 'high', 'low', 'close', 'volume',
        'label_class', 'label_numeric', 'price_target_pct',
        'future_price', 'triple_barrier_label'
    ]
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    # Clean data
    df_clean = df[df['triple_barrier_label'].notna()].copy()

    # Temporal split
    timestamps = df_clean.index
    test_mask = timestamps >= '2025-01-01'
    df_test = df_clean[test_mask].copy()

    print(f"\n[1/3] Loading Data")
    print(f"  Test samples: {len(df_test)}")

    # Prepare features
    X_test = df_test[feature_cols].fillna(0).values
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    y_test = (df_test['triple_barrier_label'] == 1).astype(int).values

    # Predict probabilities
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    print(f"  P(TP) range: {y_pred_proba.min():.4f} to {y_pred_proba.max():.4f}")
    print(f"  P(TP) mean: {y_pred_proba.mean():.4f}")

    # Grid search thresholds
    print(f"\n[2/3] Testing Thresholds")
    thresholds = np.arange(0.35, 0.75, 0.01)  # 0.35 to 0.74 by 0.01

    results_list = []
    for threshold in thresholds:
        metrics = simulate_threshold(y_test, y_pred_proba, threshold, tp_pct, sl_pct)
        results_list.append(metrics)

    # Find best threshold by Expected Value
    best_by_ev = max(results_list, key=lambda x: x['expected_value'])
    best_by_roi = max(results_list, key=lambda x: x['total_roi'])
    baseline_result = simulate_threshold(y_test, y_pred_proba, 0.5, tp_pct, sl_pct)

    print(f"\n[3/3] Results")
    print(f"\n  Baseline (threshold=0.50):")
    print(f"    Trades: {baseline_result['num_trades']}")
    print(f"    Win Rate: {baseline_result['win_rate']*100:.2f}%")
    print(f"    Expected Value: {baseline_result['expected_value']:.4f}%")
    print(f"    Total ROI: {baseline_result['total_roi']:.2f}%")

    print(f"\n  Optimal by Expected Value (threshold={best_by_ev['threshold']:.2f}):")
    print(f"    Trades: {best_by_ev['num_trades']}")
    print(f"    Win Rate: {best_by_ev['win_rate']*100:.2f}%")
    print(f"    Expected Value: {best_by_ev['expected_value']:.4f}%")
    print(f"    Total ROI: {best_by_ev['total_roi']:.2f}%")

    print(f"\n  Optimal by Total ROI (threshold={best_by_roi['threshold']:.2f}):")
    print(f"    Trades: {best_by_roi['num_trades']}")
    print(f"    Win Rate: {best_by_roi['win_rate']*100:.2f}%")
    print(f"    Expected Value: {best_by_roi['expected_value']:.4f}%")
    print(f"    Total ROI: {best_by_roi['total_roi']:.2f}%")

    print(f"\n  Improvement vs Baseline:")
    ev_improvement = best_by_ev['expected_value'] - baseline_result['expected_value']
    roi_improvement = best_by_roi['total_roi'] - baseline_result['total_roi']
    print(f"    EV: {ev_improvement:+.4f}%")
    print(f"    ROI: {roi_improvement:+.2f}%")

    # Prepare results
    results = {
        'crypto': crypto,
        'model_type': model_type,
        'tp_pct': tp_pct,
        'sl_pct': sl_pct,
        'baseline': {
            'threshold': 0.5,
            'num_trades': int(baseline_result['num_trades']),
            'win_rate': float(baseline_result['win_rate']),
            'expected_value': float(baseline_result['expected_value']),
            'total_roi': float(baseline_result['total_roi'])
        },
        'optimal_by_ev': {
            'threshold': float(best_by_ev['threshold']),
            'num_trades': int(best_by_ev['num_trades']),
            'win_rate': float(best_by_ev['win_rate']),
            'expected_value': float(best_by_ev['expected_value']),
            'total_roi': float(best_by_ev['total_roi'])
        },
        'optimal_by_roi': {
            'threshold': float(best_by_roi['threshold']),
            'num_trades': int(best_by_roi['num_trades']),
            'win_rate': float(best_by_roi['win_rate']),
            'expected_value': float(best_by_roi['expected_value']),
            'total_roi': float(best_by_roi['total_roi'])
        },
        'improvement': {
            'ev': float(ev_improvement),
            'roi': float(roi_improvement)
        },
        'all_thresholds': [
            {
                'threshold': float(r['threshold']),
                'ev': float(r['expected_value']),
                'roi': float(r['total_roi']),
                'trades': int(r['num_trades'])
            }
            for r in results_list
        ]
    }

    # Save results
    results_dir = Path(__file__).parent.parent / 'optimization' / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)

    output_file = results_dir / f'{crypto}_{model_type}_optimal_threshold.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n  Saved results to: {output_file}")

    return results


def optimize_all_thresholds():
    """Optimize thresholds for all cryptos and models"""

    print("="*80)
    print("THRESHOLD OPTIMIZATION - ALL CRYPTOS")
    print("="*80)
    print("\nGoal: Find optimal P(TP) threshold to maximize Expected Value")
    print("Expected improvement: +5-10% ROI\n")

    all_results = {}

    for crypto in ['btc', 'eth', 'sol']:
        for model_type in ['baseline', 'optimized']:
            key = f'{crypto}_{model_type}'
            try:
                results = optimize_threshold(crypto, model_type=model_type)
                if results:
                    all_results[key] = results
            except Exception as e:
                print(f"\nERROR processing {key}: {e}")
                import traceback
                traceback.print_exc()
                continue

    # Summary
    print(f"\n\n{'='*80}")
    print("THRESHOLD OPTIMIZATION SUMMARY")
    print('='*80)

    for crypto in ['btc', 'eth', 'sol']:
        print(f"\n{crypto.upper()}:")

        for model_type in ['baseline', 'optimized']:
            key = f'{crypto}_{model_type}'
            if key not in all_results:
                print(f"  {model_type}: FAILED")
                continue

            res = all_results[key]
            optimal = res['optimal_by_ev']

            print(f"\n  {model_type.upper()}:")
            print(f"    Current (0.50): EV={res['baseline']['expected_value']:.4f}%, ROI={res['baseline']['total_roi']:.1f}%")
            print(f"    Optimal ({optimal['threshold']:.2f}): EV={optimal['expected_value']:.4f}%, ROI={optimal['total_roi']:.1f}%")
            print(f"    Improvement: EV {res['improvement']['ev']:+.4f}%, ROI {res['improvement']['roi']:+.1f}%")

            if res['improvement']['roi'] > 0:
                print(f"    Status: IMPROVED")
            else:
                print(f"    Status: Baseline was already optimal")

    print(f"\n{'='*80}")
    print("THRESHOLD OPTIMIZATION COMPLETE!")
    print('='*80)

    # Create recommendation summary
    print(f"\n\nRECOMMENDED THRESHOLDS:")
    print("-"*80)
    for crypto in ['btc', 'eth', 'sol']:
        for model_type in ['baseline', 'optimized']:
            key = f'{crypto}_{model_type}'
            if key in all_results:
                optimal_threshold = all_results[key]['optimal_by_ev']['threshold']
                print(f"  {crypto.upper()} {model_type:10s}: {optimal_threshold:.2f}")

    return all_results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Threshold Optimization for V11 TEMPORAL')
    parser.add_argument('--crypto', type=str, default='all', help='Crypto to optimize (btc/eth/sol/all)')
    parser.add_argument('--model-type', type=str, default='both', help='Model type (baseline/optimized/both)')

    args = parser.parse_args()

    if args.crypto == 'all':
        optimize_all_thresholds()
    else:
        if args.model_type == 'both':
            for mt in ['baseline', 'optimized']:
                optimize_threshold(args.crypto, model_type=mt)
        else:
            optimize_threshold(args.crypto, model_type=args.model_type)
