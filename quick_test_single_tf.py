"""
OPTION A: Quick Single-Timeframe Test
======================================
Validate that the V10 pipeline works end-to-end before full multi-TF implementation

Test Scope:
- Single timeframe (1d)
- Single crypto (BTC)
- Basic XGBoost classifier
- V9 features (base + temporal)
- 7-day lookahead labels

This is a SANITY CHECK to ensure:
1. Data loading works
2. Feature calculation works
3. Label generation works
4. Model training works
5. Can produce predictions with reasonable accuracy

Expected runtime: 2-5 minutes
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from data.data_manager_multi_tf import get_dataframe
from features.base_indicators import calculate_base_indicators
from features.temporal_features import calculate_temporal_features
from features.labels import generate_labels


def quick_test():
    """Quick single-TF test on BTC 1d data"""

    print("=" * 80)
    print("OPTION A: QUICK SINGLE-TIMEFRAME TEST")
    print("=" * 80)

    # Test parameters
    crypto = 'btc'
    timeframe = '1d'

    print(f"\nTest Configuration:")
    print(f"  Crypto: {crypto.upper()}")
    print(f"  Timeframe: {timeframe}")
    print(f"  Features: Base (30) + Temporal (49) = 79 total")
    print(f"  Lookahead: 7 days")
    print(f"  Model: XGBoost Classifier")

    # Step 1: Load data
    print(f"\n{'='*80}")
    print("[1/5] Loading Data")
    print('='*80)

    df = get_dataframe(crypto, timeframe)
    if df is None:
        print(f"ERROR: Failed to load {crypto} {timeframe} data")
        return False

    print(f"  + Loaded {len(df)} rows")
    print(f"  + Date range: {df.index[0]} to {df.index[-1]}")
    print(f"  + Columns: {list(df.columns)}")

    # Step 2: Calculate features
    print(f"\n{'='*80}")
    print("[2/5] Calculating Features")
    print('='*80)

    initial_cols = len(df.columns)

    # Base indicators
    print(f"  Calculating base indicators...")
    df = calculate_base_indicators(df, timeframe)
    base_cols = len(df.columns) - initial_cols
    print(f"  + Added {base_cols} base features")

    # Temporal features
    print(f"  Calculating temporal features...")
    cols_before_temporal = len(df.columns)
    df = calculate_temporal_features(df, timeframe)
    temporal_cols = len(df.columns) - cols_before_temporal
    print(f"  + Added {temporal_cols} temporal features")

    total_features = base_cols + temporal_cols
    print(f"\n  Total features: {total_features}")
    print(f"  Total columns: {len(df.columns)}")

    # Step 3: Generate labels
    print(f"\n{'='*80}")
    print("[3/5] Generating Labels")
    print('='*80)

    df, label_stats = generate_labels(df, timeframe, crypto)

    print(f"  Lookahead: 7 candles (7 days)")
    print(f"\n  Label Distribution:")
    print(f"    BUY:  {label_stats['classification']['counts']['BUY']:4d} ({label_stats['classification']['percentages']['BUY']:5.1f}%)")
    print(f"    HOLD: {label_stats['classification']['counts']['HOLD']:4d} ({label_stats['classification']['percentages']['HOLD']:5.1f}%)")
    print(f"    SELL: {label_stats['classification']['counts']['SELL']:4d} ({label_stats['classification']['percentages']['SELL']:5.1f}%)")
    print(f"    Total: {label_stats['classification']['counts']['total']} valid labels")

    print(f"\n  Price Target Statistics:")
    print(f"    Mean:   {label_stats['regression']['mean']:+6.2f}%")
    print(f"    Median: {label_stats['regression']['median']:+6.2f}%")
    print(f"    Std:    {label_stats['regression']['std']:6.2f}%")
    print(f"    Range:  {label_stats['regression']['min']:+6.2f}% to {label_stats['regression']['max']:+6.2f}%")

    # Step 4: Prepare data for training
    print(f"\n{'='*80}")
    print("[4/5] Preparing Training Data")
    print('='*80)

    # Get feature columns (exclude OHLCV and label columns)
    exclude_cols = ['open', 'high', 'low', 'close', 'volume',
                    'label_class', 'label_numeric', 'price_target_pct', 'future_price']
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    # Remove rows with NaN labels
    df_clean = df[df['label_class'].notna()].copy()

    print(f"  Feature columns: {len(feature_cols)}")
    print(f"  Valid rows (after removing NaN labels): {len(df_clean)}")

    # Handle any remaining NaN in features (fill with 0)
    X = df_clean[feature_cols].fillna(0).values
    y = df_clean['label_numeric'].values

    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Unique labels: {np.unique(y)} (SELL=-1, HOLD=0, BUY=1)")

    # Check for any remaining NaN or inf
    nan_count = np.isnan(X).sum()
    inf_count = np.isinf(X).sum()
    if nan_count > 0 or inf_count > 0:
        print(f"  WARNING: Found {nan_count} NaN and {inf_count} inf values")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        print(f"  Fixed: Replaced with 0")

    # Step 5: Train and evaluate model
    print(f"\n{'='*80}")
    print("[5/5] Training Model")
    print('='*80)

    # Use TimeSeriesSplit for walk-forward validation
    n_splits = 5
    tscv = TimeSeriesSplit(n_splits=n_splits)

    print(f"  Using TimeSeriesSplit with {n_splits} splits")
    print(f"  Model: XGBoost Classifier (V9 baseline params)")

    # V9 baseline parameters (conservative, anti-overfitting)
    xgb_params = {
        'objective': 'multi:softmax',
        'num_class': 3,
        'max_depth': 3,
        'learning_rate': 0.01,
        'n_estimators': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 5,
        'min_child_weight': 3,
        'reg_alpha': 1.0,
        'reg_lambda': 1.0,
        'random_state': 42,
        'tree_method': 'hist',
        'eval_metric': 'mlogloss'
    }

    # Store results from each split
    train_accuracies = []
    val_accuracies = []

    print(f"\n  Training {n_splits} models with walk-forward validation...\n")

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Convert labels to 0, 1, 2 (XGBoost expects 0-indexed)
        y_train_xgb = y_train + 1  # -1,0,1 -> 0,1,2
        y_val_xgb = y_val + 1

        # Train model
        model = xgb.XGBClassifier(**xgb_params, verbosity=0)
        model.fit(X_train, y_train_xgb)

        # Predictions
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)

        # Convert back to -1,0,1
        y_train_pred = y_train_pred - 1
        y_val_pred = y_val_pred - 1

        # Accuracies
        train_acc = accuracy_score(y_train, y_train_pred)
        val_acc = accuracy_score(y_val, y_val_pred)

        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(f"  Fold {fold}:")
        print(f"    Train: {len(X_train):4d} rows | Accuracy: {train_acc:.4f}")
        print(f"    Val:   {len(X_val):4d} rows | Accuracy: {val_acc:.4f}")
        print(f"    Overfitting gap: {(train_acc - val_acc)*100:+.2f}%")

    # Final summary
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print('='*80)

    avg_train_acc = np.mean(train_accuracies)
    avg_val_acc = np.mean(val_accuracies)
    overfitting = (avg_train_acc - avg_val_acc) * 100

    print(f"\nAverage Performance Across {n_splits} Folds:")
    print(f"  Training Accuracy:   {avg_train_acc:.4f} ({avg_train_acc*100:.2f}%)")
    print(f"  Validation Accuracy: {avg_val_acc:.4f} ({avg_val_acc*100:.2f}%)")
    print(f"  Overfitting Gap:     {overfitting:+.2f}%")

    # Train final model on all data for detailed analysis
    print(f"\n{'='*80}")
    print("FINAL MODEL ANALYSIS")
    print('='*80)

    # Use last 20% as test set
    test_size = int(len(X) * 0.2)
    X_train_final = X[:-test_size]
    X_test_final = X[-test_size:]
    y_train_final = y[:-test_size]
    y_test_final = y[-test_size:]

    y_train_final_xgb = y_train_final + 1
    y_test_final_xgb = y_test_final + 1

    model_final = xgb.XGBClassifier(**xgb_params, verbosity=0)
    model_final.fit(X_train_final, y_train_final_xgb)

    y_pred_final = model_final.predict(X_test_final) - 1

    print(f"\nTest Set Performance (last 20% = {test_size} rows):")
    print(f"  Accuracy: {accuracy_score(y_test_final, y_pred_final):.4f}")

    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_test_final, y_pred_final, labels=[-1, 0, 1])
    print(f"              Predicted")
    print(f"              SELL  HOLD   BUY")
    print(f"Actual SELL  {cm[0][0]:5d} {cm[0][1]:5d} {cm[0][2]:5d}")
    print(f"       HOLD  {cm[1][0]:5d} {cm[1][1]:5d} {cm[1][2]:5d}")
    print(f"       BUY   {cm[2][0]:5d} {cm[2][1]:5d} {cm[2][2]:5d}")

    print(f"\nClassification Report:")
    target_names = ['SELL', 'HOLD', 'BUY']
    print(classification_report(y_test_final, y_pred_final, target_names=target_names))

    # Feature importance
    print(f"\n{'='*80}")
    print("TOP 20 MOST IMPORTANT FEATURES")
    print('='*80)

    feature_importance = model_final.feature_importances_
    feature_names = np.array(feature_cols)

    # Sort by importance
    sorted_idx = np.argsort(feature_importance)[::-1]

    print(f"\n  Rank  Feature                                    Importance")
    print(f"  {'-'*60}")
    for i, idx in enumerate(sorted_idx[:20], 1):
        print(f"  {i:3d}   {feature_names[idx]:40s}   {feature_importance[idx]:.4f}")

    # Final verdict
    print(f"\n{'='*80}")
    print("TEST VERDICT")
    print('='*80)

    success_criteria = [
        ("Data loaded successfully", True),
        ("Features calculated without errors", total_features == 79),
        ("Labels generated correctly", label_stats['valid_count'] > 0),
        ("Model trained successfully", avg_val_acc > 0),
        ("Validation accuracy > 40%", avg_val_acc > 0.40),
        ("Overfitting < 15%", overfitting < 15)
    ]

    all_passed = True
    for criterion, passed in success_criteria:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {criterion}")
        if not passed:
            all_passed = False

    print(f"\n{'='*80}")
    if all_passed:
        print("TEST PASSED - Pipeline is working correctly!")
        print("Ready to proceed with T8-T10 (multi-TF implementation)")
    else:
        print("TEST FAILED - Fix issues before proceeding")
    print('='*80)

    return all_passed


if __name__ == '__main__':
    success = quick_test()
    sys.exit(0 if success else 1)
