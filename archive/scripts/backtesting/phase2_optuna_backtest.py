"""
PHASE 2.5 BACKTEST - Test Optuna Model en ROI
==============================================
Compare Phase 1 (feature-selected) vs Optuna model for SOL

PHASE 1 SOL:
- Model: sol_v11_feature_selected_top50.joblib (59.01% accuracy)
- Threshold: 0.35
- ROI: +64.48%

OPTUNA SOL:
- Model: sol_v11_optimized.joblib (63.06% accuracy, +7.43%)
- Threshold: 0.35 (to test)
- ROI: ??? (TO TEST!)
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import json


def simulate_trading_optuna(crypto: str, model_type: str, initial_capital=1000.0,
                            tp_pct=1.5, sl_pct=0.75, threshold=0.35):
    """
    Simulate trading with different model types

    Args:
        crypto: 'btc', 'eth', or 'sol'
        model_type: 'feature_selected', 'optimized', or 'baseline'
        initial_capital: Starting capital ($)
        tp_pct: Take profit (%)
        sl_pct: Stop loss (%)
        threshold: Probability threshold

    Returns:
        Results dictionary and trades dataframe
    """

    # Load model
    if model_type == 'feature_selected':
        model_file = Path(__file__).parent.parent / 'models' / f'{crypto}_v11_feature_selected_top50.joblib'
        features_file = Path(__file__).parent.parent / 'optimization' / 'results' / f'{crypto}_selected_features_top50.json'
    elif model_type == 'optimized':
        model_file = Path(__file__).parent.parent / 'models' / f'{crypto}_v11_optimized.joblib'
        features_file = None  # Uses all features
    else:  # baseline
        model_file = Path(__file__).parent.parent / 'models' / f'{crypto}_v11_classifier.joblib'
        features_file = None

    if not model_file.exists():
        raise FileNotFoundError(f"Model not found: {model_file}")

    model = joblib.load(model_file)

    # Load selected features if needed
    if features_file and features_file.exists():
        with open(features_file) as f:
            feature_data = json.load(f)
            selected_features = feature_data['selected_feature_names']
    else:
        selected_features = None

    # Load data
    cache_file = Path(__file__).parent.parent / 'data' / 'cache' / f'{crypto}_multi_tf_merged.csv'
    df = pd.read_csv(cache_file, index_col=0, parse_dates=True)

    # Prepare features
    exclude_cols = ['open', 'high', 'low', 'close', 'volume',
                   'label_class', 'label_numeric', 'price_target_pct',
                   'future_price', 'triple_barrier_label']

    if selected_features:
        feature_cols = selected_features
    else:
        feature_cols = [col for col in df.columns if col not in exclude_cols]

    # TEMPORAL SPLIT: Test on 2025+
    df_clean = df[df['triple_barrier_label'].notna()].copy()
    timestamps = df_clean.index
    test_mask = timestamps >= '2025-01-01'
    df_test = df_clean[test_mask].copy()

    print(f"  Test samples: {len(df_test)}")

    # Prepare test features
    X_test = df_test[feature_cols].fillna(0).values
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    y_test = df_test['label_numeric'].values

    # Predict probabilities
    probs = model.predict_proba(X_test)[:, 1]
    signals = (probs > threshold).astype(int)

    # Backtest
    capital = initial_capital
    trades = []

    for i, (idx, row) in enumerate(df_test.iterrows()):
        if signals[i] == 1:  # BUY signal
            actual_label = y_test[i]
            prob = probs[i]

            if actual_label == 1:  # TP
                pnl_pct = tp_pct
                outcome = 'TP'
            else:  # SL
                pnl_pct = -sl_pct
                outcome = 'SL'

            pnl = capital * (pnl_pct / 100)
            capital += pnl

            trades.append({
                'timestamp': idx,
                'signal_prob': prob,
                'outcome': outcome,
                'pnl_pct': pnl_pct,
                'pnl_usd': pnl,
                'capital': capital
            })

    # Calculate metrics
    trades_df = pd.DataFrame(trades)

    if len(trades_df) > 0:
        total_trades = len(trades_df)
        tp_trades = (trades_df['outcome'] == 'TP').sum()
        sl_trades = (trades_df['outcome'] == 'SL').sum()
        win_rate = tp_trades / total_trades if total_trades > 0 else 0

        final_capital = capital
        roi_pct = ((final_capital - initial_capital) / initial_capital) * 100

        # Accuracy
        y_pred = signals
        accuracy = (y_pred == y_test).mean()
    else:
        total_trades = 0
        tp_trades = 0
        sl_trades = 0
        win_rate = 0
        final_capital = initial_capital
        roi_pct = 0
        accuracy = 0

    results = {
        'model_type': model_type,
        'threshold': threshold,
        'initial_capital': initial_capital,
        'final_capital': final_capital,
        'roi_pct': roi_pct,
        'total_trades': total_trades,
        'tp_trades': tp_trades,
        'sl_trades': sl_trades,
        'win_rate': win_rate * 100,
        'accuracy': accuracy * 100,
        'tp_pct': tp_pct,
        'sl_pct': sl_pct
    }

    return results, trades_df


def main():
    print("=" * 70)
    print("PHASE 2.5: OPTUNA MODEL BACKTEST (SOL)")
    print("=" * 70)
    print()

    crypto = 'sol'

    # Test 3 configurations
    configs = [
        ('baseline', 'Baseline (all features)'),
        ('feature_selected', 'Phase 1 (top 50 features)'),
        ('optimized', 'Optuna (all features, tuned hyperparams)')
    ]

    results_summary = []

    for model_type, description in configs:
        print(f"\n{description}")
        print("-" * 70)

        try:
            results, trades_df = simulate_trading_optuna(
                crypto=crypto,
                model_type=model_type,
                threshold=0.35,  # Use same threshold for fair comparison
                tp_pct=1.5,
                sl_pct=0.75
            )

            results_summary.append(results)

            print(f"  Model: {model_type}")
            print(f"  Accuracy: {results['accuracy']:.2f}%")
            print(f"  Total Trades: {results['total_trades']}")
            print(f"  Win Rate: {results['win_rate']:.2f}%")
            print(f"  TP/SL: {results['tp_trades']}/{results['sl_trades']}")
            print(f"  ROI: {results['roi_pct']:+.2f}%")
            print(f"  Capital: ${results['initial_capital']:.2f} -> ${results['final_capital']:.2f}")

        except FileNotFoundError as e:
            print(f"  ERROR: {e}")

    # Comparison
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print()

    if len(results_summary) >= 2:
        baseline = results_summary[0]
        phase1 = results_summary[1]

        print(f"BASELINE vs PHASE 1:")
        print(f"  Accuracy: {baseline['accuracy']:.2f}% -> {phase1['accuracy']:.2f}% ({phase1['accuracy']-baseline['accuracy']:+.2f}%)")
        print(f"  ROI: {baseline['roi_pct']:+.2f}% -> {phase1['roi_pct']:+.2f}% ({phase1['roi_pct']-baseline['roi_pct']:+.2f}%)")
        print(f"  Trades: {baseline['total_trades']} -> {phase1['total_trades']} ({phase1['total_trades']-baseline['total_trades']:+d})")
        print()

        if len(results_summary) >= 3:
            optuna = results_summary[2]

            print(f"PHASE 1 vs OPTUNA:")
            print(f"  Accuracy: {phase1['accuracy']:.2f}% -> {optuna['accuracy']:.2f}% ({optuna['accuracy']-phase1['accuracy']:+.2f}%)")
            print(f"  ROI: {phase1['roi_pct']:+.2f}% -> {optuna['roi_pct']:+.2f}% ({optuna['roi_pct']-phase1['roi_pct']:+.2f}%)")
            print(f"  Trades: {phase1['total_trades']} -> {optuna['total_trades']} ({optuna['total_trades']-phase1['total_trades']:+d})")
            print()

            print(f"BASELINE vs OPTUNA:")
            print(f"  Accuracy: {baseline['accuracy']:.2f}% -> {optuna['accuracy']:.2f}% ({optuna['accuracy']-baseline['accuracy']:+.2f}%)")
            print(f"  ROI: {baseline['roi_pct']:+.2f}% -> {optuna['roi_pct']:+.2f}% ({optuna['roi_pct']-baseline['roi_pct']:+.2f}%)")
            print(f"  Trades: {baseline['total_trades']} -> {optuna['total_trades']} ({optuna['total_trades']-baseline['total_trades']:+d})")
            print()

            # Recommendation
            print("=" * 70)
            print("RECOMMENDATION")
            print("=" * 70)
            print()

            best_roi = max(results_summary, key=lambda x: x['roi_pct'])
            best_accuracy = max(results_summary, key=lambda x: x['accuracy'])

            print(f"Best ROI: {best_roi['model_type']} ({best_roi['roi_pct']:+.2f}%)")
            print(f"Best Accuracy: {best_accuracy['model_type']} ({best_accuracy['accuracy']:.2f}%)")
            print()

            if best_roi['model_type'] == 'optimized':
                print("OPTUNA MODEL IS BETTER! Use sol_v11_optimized.joblib")
                improvement = optuna['roi_pct'] - phase1['roi_pct']
                print(f"  -> +{improvement:.2f}% ROI improvement over Phase 1")
            elif best_roi['model_type'] == 'feature_selected':
                print("PHASE 1 MODEL IS BETTER! Keep sol_v11_feature_selected_top50.joblib")
                improvement = phase1['roi_pct'] - optuna['roi_pct']
                print(f"  -> Phase 1 is +{improvement:.2f}% better than Optuna")
            else:
                print("BASELINE IS BEST (surprising!)")

    print()
    print("=" * 70)


if __name__ == '__main__':
    main()
