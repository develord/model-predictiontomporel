"""
WALK-FORWARD VALIDATION - V11 TEMPORAL
=======================================
Test la robustesse de Phase 1 sur différentes périodes temporelles.

EXPANDING WINDOW APPROACH:
- Period 1: Train 2021-2022 → Test 2023
- Period 2: Train 2021-2023 → Test 2024
- Period 3: Train 2021-2024 → Test 2025 (configuration actuelle)

OBJECTIF:
Vérifier si Phase 1 (threshold + feature selection) est robuste
ou si les performances sont spécifiques aux données 2025.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
from datetime import datetime
from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

# ============================================================================
# CONFIGURATION
# ============================================================================

CRYPTOS = ['btc', 'eth', 'sol']

# Phase 1 Configuration (from Phase 1 results)
PHASE1_CONFIG = {
    'btc': {
        'use_feature_selection': False,  # All 237 features
        'n_features': None,
        'optimal_threshold': 0.37
    },
    'eth': {
        'use_feature_selection': False,  # All 348 features
        'n_features': None,
        'optimal_threshold': 0.35
    },
    'sol': {
        'use_feature_selection': True,   # Top 50 features
        'n_features': 50,
        'optimal_threshold': 0.35
    }
}

# Trading Parameters
TP_PCT = 1.5
SL_PCT = 0.75

# Periods for Walk-Forward
PERIODS = [
    {
        'name': 'Period_1',
        'train_start': '2021-01-01',
        'train_end': '2023-01-01',
        'test_start': '2023-01-01',
        'test_end': '2024-01-01',
        'description': 'Train 2021-2022, Test 2023'
    },
    {
        'name': 'Period_2',
        'train_start': '2021-01-01',
        'train_end': '2024-01-01',
        'test_start': '2024-01-01',
        'test_end': '2025-01-01',
        'description': 'Train 2021-2023, Test 2024'
    },
    {
        'name': 'Period_3',
        'train_start': '2021-01-01',
        'train_end': '2025-01-01',
        'test_start': '2025-01-01',
        'test_end': '2026-01-01',
        'description': 'Train 2021-2024, Test 2025 (Current)'
    }
]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_data(crypto: str):
    """Load merged data for crypto."""
    base_dir = Path(__file__).parent.parent
    data_file = base_dir / 'data' / 'cache' / f'{crypto}_multi_tf_merged.csv'

    print(f"  Loading data: {data_file}")
    df = pd.read_csv(data_file)

    # Convert date to timestamp
    df['timestamp'] = pd.to_datetime(df['date'])

    # Filter to binary labels only (0=SL, 1=TP)
    df = df[df['triple_barrier_label'].notna()].copy()
    df = df[df['label_numeric'].isin([0, 1])].copy()

    print(f"  Loaded {len(df)} samples (binary labels only)")
    return df

def split_temporal(df, train_start, train_end, test_start, test_end):
    """Split data temporally."""
    train_mask = (df['timestamp'] >= train_start) & (df['timestamp'] < train_end)
    test_mask = (df['timestamp'] >= test_start) & (df['timestamp'] < test_end)

    df_train = df[train_mask].copy()
    df_test = test_mask = df[test_mask].copy()

    return df_train, df_test

def prepare_features(df):
    """Extract features and labels."""
    exclude_cols = [
        'timestamp', 'date', 'triple_barrier_label', 'label_numeric',
        'label_class', 'price_target_pct'
    ]

    feature_cols = [col for col in df.columns if col not in exclude_cols]

    X = df[feature_cols].copy()
    y = df['label_numeric'].copy()

    # Handle inf/nan
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())

    return X, y, feature_cols

def train_model(X_train, y_train, crypto: str, use_feature_selection: bool, n_features: int = None):
    """Train XGBoost model with optional feature selection."""

    # Base XGBoost parameters
    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'random_state': 42,
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    }

    base_model = xgb.XGBClassifier(**xgb_params)

    if use_feature_selection and n_features:
        print(f"    Applying RFE feature selection: {X_train.shape[1]} -> {n_features} features")
        rfe = RFE(estimator=base_model, n_features_to_select=n_features, step=10)
        rfe.fit(X_train, y_train)

        selected_features = X_train.columns[rfe.support_].tolist()
        X_train_selected = X_train[selected_features]

        # Retrain on selected features
        final_model = xgb.XGBClassifier(**xgb_params)
        final_model.fit(X_train_selected, y_train)

        return final_model, selected_features
    else:
        print(f"    Training on all {X_train.shape[1]} features")
        base_model.fit(X_train, y_train)
        return base_model, X_train.columns.tolist()

def optimize_threshold(model, X_val, y_val, feature_cols, thresholds=None):
    """Find optimal threshold by ROI simulation on validation set."""

    if thresholds is None:
        thresholds = np.arange(0.20, 0.55, 0.05)

    # Get predictions
    X_val_features = X_val[feature_cols]
    y_proba = model.predict_proba(X_val_features)[:, 1]

    best_roi = -np.inf
    best_threshold = 0.5

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)

        # Simulate trading on validation set
        roi = calculate_roi(y_val, y_pred)

        if roi > best_roi:
            best_roi = roi
            best_threshold = threshold

    print(f"    Optimal threshold: {best_threshold:.2f} (ROI: {best_roi:.2f}%)")
    return best_threshold

def calculate_roi(y_true, y_pred):
    """Calculate ROI% based on predictions."""
    capital = 1000.0

    for true_label, pred_label in zip(y_true, y_pred):
        if pred_label == 1:  # BUY signal
            if true_label == 1:  # TP hit
                capital *= (1 + TP_PCT / 100)
            else:  # SL hit
                capital *= (1 - SL_PCT / 100)

    roi = ((capital - 1000) / 1000) * 100
    return roi

def backtest_trading(model, X_test, y_test, feature_cols, threshold, initial_capital=1000.0):
    """Backtest trading strategy with model."""

    # Get predictions
    X_test_features = X_test[feature_cols]
    y_proba = model.predict_proba(X_test_features)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    # Simulate trading
    capital = initial_capital
    trades = []

    for i, (true_label, pred_label, proba) in enumerate(zip(y_test, y_pred, y_proba)):
        if pred_label == 1:  # BUY signal
            entry_capital = capital

            if true_label == 1:  # TP hit
                pnl = entry_capital * (TP_PCT / 100)
                capital += pnl
                result = 'TP'
            else:  # SL hit
                pnl = -entry_capital * (SL_PCT / 100)
                capital += pnl
                result = 'SL'

            trades.append({
                'result': result,
                'pnl': pnl,
                'capital': capital,
                'proba': proba
            })

    # Calculate metrics
    n_trades = len(trades)

    if n_trades > 0:
        wins = [t for t in trades if t['result'] == 'TP']
        win_rate = (len(wins) / n_trades) * 100

        # ROI
        roi = ((capital - initial_capital) / initial_capital) * 100

        # Sharpe Ratio (simplified)
        returns = [t['pnl'] / initial_capital for t in trades]
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252)  # Annualized
        else:
            sharpe = 0.0

        # Max Drawdown
        peak = initial_capital
        max_dd = 0
        for t in trades:
            if t['capital'] > peak:
                peak = t['capital']
            dd = ((peak - t['capital']) / peak) * 100
            if dd > max_dd:
                max_dd = dd
    else:
        win_rate = 0
        roi = 0
        sharpe = 0
        max_dd = 0

    results = {
        'n_trades': n_trades,
        'win_rate': win_rate,
        'roi': roi,
        'final_capital': capital,
        'sharpe': sharpe,
        'max_drawdown': max_dd
    }

    return results

# ============================================================================
# MAIN WALK-FORWARD VALIDATION
# ============================================================================

def run_walk_forward():
    """Run walk-forward validation for all cryptos and periods."""

    print("=" * 80)
    print("WALK-FORWARD VALIDATION - V11 TEMPORAL")
    print("=" * 80)
    print()
    print("Configuration: Phase 1 (Threshold Optimization + Feature Selection)")
    print()

    all_results = {}

    for crypto in CRYPTOS:
        print(f"\n{'=' * 80}")
        print(f"CRYPTO: {crypto.upper()}")
        print(f"{'=' * 80}")

        # Load full data
        df = load_data(crypto)

        config = PHASE1_CONFIG[crypto]
        crypto_results = {}

        for period in PERIODS:
            print(f"\n{period['name']}: {period['description']}")
            print("-" * 80)

            # Split data
            df_train, df_test = split_temporal(
                df,
                period['train_start'],
                period['train_end'],
                period['test_start'],
                period['test_end']
            )

            print(f"  Train: {len(df_train)} samples ({period['train_start']} to {period['train_end']})")
            print(f"  Test:  {len(df_test)} samples ({period['test_start']} to {period['test_end']})")

            if len(df_train) == 0 or len(df_test) == 0:
                print("  [SKIP] Not enough data for this period")
                continue

            # Prepare features
            X_train, y_train, feature_cols = prepare_features(df_train)
            X_test, y_test, _ = prepare_features(df_test)

            # Split train into train/val for threshold optimization (80/20)
            split_idx = int(len(X_train) * 0.8)
            X_train_sub = X_train.iloc[:split_idx]
            y_train_sub = y_train.iloc[:split_idx]
            X_val = X_train.iloc[split_idx:]
            y_val = y_train.iloc[split_idx:]

            # Train model
            print(f"  Training model...")
            model, selected_features = train_model(
                X_train_sub,
                y_train_sub,
                crypto,
                config['use_feature_selection'],
                config['n_features']
            )

            # Optimize threshold on validation set
            print(f"  Optimizing threshold on validation set...")
            optimal_threshold = optimize_threshold(model, X_val, y_val, selected_features)

            # Backtest on test set
            print(f"  Backtesting on test set...")
            test_results = backtest_trading(
                model,
                X_test,
                y_test,
                selected_features,
                optimal_threshold
            )

            print(f"\n  Results:")
            print(f"    Threshold: {optimal_threshold:.2f}")
            print(f"    Trades:    {test_results['n_trades']}")
            print(f"    Win Rate:  {test_results['win_rate']:.2f}%")
            print(f"    ROI:       {test_results['roi']:+.2f}%")
            print(f"    Sharpe:    {test_results['sharpe']:.2f}")
            print(f"    Max DD:    {test_results['max_drawdown']:.2f}%")
            print(f"    Capital:   ${test_results['final_capital']:.0f}")

            # Store results
            crypto_results[period['name']] = {
                'threshold': optimal_threshold,
                'n_features': len(selected_features),
                **test_results
            }

        all_results[crypto] = crypto_results

    # ========================================================================
    # COMPARATIVE ANALYSIS
    # ========================================================================

    print("\n" + "=" * 80)
    print("COMPARATIVE ANALYSIS")
    print("=" * 80)

    for crypto in CRYPTOS:
        print(f"\n{crypto.upper()}:")
        print("-" * 80)

        if crypto not in all_results:
            continue

        results = all_results[crypto]

        # Table header
        print(f"{'Period':<15} {'Threshold':<12} {'Trades':<8} {'Win Rate':<10} {'ROI':<12} {'Sharpe':<10} {'Max DD':<10}")
        print("-" * 80)

        for period_name in ['Period_1', 'Period_2', 'Period_3']:
            if period_name not in results:
                continue

            r = results[period_name]
            print(f"{period_name:<15} {r['threshold']:<12.2f} {r['n_trades']:<8} {r['win_rate']:<10.2f} {r['roi']:+<12.2f} {r['sharpe']:<10.2f} {r['max_drawdown']:<10.2f}")

    # Portfolio Summary
    print("\n" + "=" * 80)
    print("PORTFOLIO SUMMARY (Equal Weight BTC+ETH+SOL)")
    print("=" * 80)

    portfolio_results = {}

    for period_name in ['Period_1', 'Period_2', 'Period_3']:
        period_rois = []
        period_trades = 0
        period_wins = 0

        for crypto in CRYPTOS:
            if crypto in all_results and period_name in all_results[crypto]:
                r = all_results[crypto][period_name]
                period_rois.append(r['roi'])
                period_trades += r['n_trades']
                period_wins += (r['n_trades'] * r['win_rate'] / 100)

        if len(period_rois) == 3:
            portfolio_roi = np.mean(period_rois)
            portfolio_win_rate = (period_wins / period_trades * 100) if period_trades > 0 else 0

            portfolio_results[period_name] = {
                'roi': portfolio_roi,
                'win_rate': portfolio_win_rate,
                'total_trades': period_trades
            }

    print(f"\n{'Period':<15} {'Portfolio ROI':<15} {'Win Rate':<12} {'Total Trades':<15}")
    print("-" * 80)

    for period_name in ['Period_1', 'Period_2', 'Period_3']:
        if period_name in portfolio_results:
            r = portfolio_results[period_name]
            period_desc = [p['description'] for p in PERIODS if p['name'] == period_name][0]
            print(f"{period_desc:<15} {r['roi']:+.2f}%{' ':<10} {r['win_rate']:.2f}%{' ':<6} {r['total_trades']:<15}")

    # ========================================================================
    # ROBUSTNESS ANALYSIS
    # ========================================================================

    print("\n" + "=" * 80)
    print("ROBUSTNESS ANALYSIS")
    print("=" * 80)

    portfolio_rois = [portfolio_results[p]['roi'] for p in portfolio_results]

    if len(portfolio_rois) >= 2:
        roi_mean = np.mean(portfolio_rois)
        roi_std = np.std(portfolio_rois)
        roi_min = np.min(portfolio_rois)
        roi_max = np.max(portfolio_rois)

        print(f"\nPortfolio ROI Statistics:")
        print(f"  Mean:   {roi_mean:+.2f}%")
        print(f"  Std:    {roi_std:.2f}%")
        print(f"  Min:    {roi_min:+.2f}%")
        print(f"  Max:    {roi_max:+.2f}%")
        print(f"  Range:  {roi_max - roi_min:.2f}%")

        # Consistency score
        consistency = (roi_std / abs(roi_mean)) * 100 if roi_mean != 0 else 0

        print(f"\n  Consistency Score: {consistency:.2f}% (lower is better)")

        if consistency < 20:
            verdict = "EXCELLENT - Very consistent across periods"
        elif consistency < 40:
            verdict = "GOOD - Reasonably consistent"
        elif consistency < 60:
            verdict = "MODERATE - Some variability"
        else:
            verdict = "POOR - High variability, possible overfitting"

        print(f"  Verdict: {verdict}")

        # Check if all periods are profitable
        all_profitable = all(roi > 0 for roi in portfolio_rois)
        print(f"\n  All Periods Profitable: {'YES' if all_profitable else 'NO'}")

        if all_profitable and consistency < 40:
            print("\n  [OK] Phase 1 is ROBUST - Good performance across all periods!")
        elif all_profitable:
            print("\n  [WARNING] Profitable but inconsistent - Monitor carefully")
        else:
            print("\n  [ALERT] Some periods unprofitable - Model may be overfitted to Period 3")

    # ========================================================================
    # SAVE RESULTS
    # ========================================================================

    output_dir = Path(__file__).parent / 'results'
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / 'walk_forward_results.json'

    with open(output_file, 'w') as f:
        json.dump({
            'crypto_results': all_results,
            'portfolio_results': portfolio_results,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)

    print(f"\n\nResults saved to: {output_file}")

    print("\n" + "=" * 80)
    print("WALK-FORWARD VALIDATION COMPLETE")
    print("=" * 80)

if __name__ == '__main__':
    run_walk_forward()
