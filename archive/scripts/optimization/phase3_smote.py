"""
PHASE 3: DATA BALANCING WITH SMOTE
===================================

Objectif: Corriger distribution shift entre train/test
Technique: SMOTE (Synthetic Minority Over-sampling Technique)

Distribution shifts actuels:
- BTC: 56% TP train -> 34.5% test (-8.5% shift)  <- PRIORITY
- SOL: 76% TP train -> 64% test (-10.4% shift)   <- PRIORITY
- ETH: 57% TP train -> 54% test (-2.6% shift)    <- LOW

Impact attendu:
- BTC: +3-5% accuracy, +6-10% ROI
- SOL: +1-2% accuracy, +5-10% ROI
- ETH: +1-3% accuracy, +3-7% ROI
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report


def load_and_prepare_data(crypto: str):
    """Load data and prepare for SMOTE"""
    print(f"\n{'='*70}")
    print(f"LOADING DATA: {crypto.upper()}")
    print('='*70)

    # Load merged data
    cache_file = Path(__file__).parent.parent / 'data' / 'cache' / f'{crypto}_multi_tf_merged.csv'
    df = pd.read_csv(cache_file, index_col=0, parse_dates=True)

    # Exclude non-feature columns
    exclude_cols = ['open', 'high', 'low', 'close', 'volume',
                   'label_class', 'label_numeric', 'price_target_pct',
                   'future_price', 'triple_barrier_label']
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    # Clean data and filter to binary labels only (0=SL, 1=TP, exclude -1)
    df_clean = df[df['triple_barrier_label'].notna()].copy()
    df_clean = df_clean[df_clean['label_numeric'].isin([0, 1])].copy()  # Filter out -1 labels

    print(f"  Filtered to binary labels (0, 1) only")

    # Temporal split
    train_mask = df_clean.index < '2025-01-01'
    test_mask = df_clean.index >= '2025-01-01'

    df_train = df_clean[train_mask].copy()
    df_test = df_clean[test_mask].copy()

    # Prepare features and labels
    X_train = df_train[feature_cols].fillna(0).values
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    y_train = df_train['label_numeric'].values

    X_test = df_test[feature_cols].fillna(0).values
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    y_test = df_test['label_numeric'].values

    # Print distribution
    train_tp_pct = (y_train == 1).sum() / len(y_train) * 100
    test_tp_pct = (y_test == 1).sum() / len(y_test) * 100
    shift = test_tp_pct - train_tp_pct

    print(f"\nOriginal Distribution:")
    print(f"  Train: {len(y_train)} samples, TP={train_tp_pct:.1f}%")
    print(f"  Test:  {len(y_test)} samples, TP={test_tp_pct:.1f}%")
    print(f"  Shift: {shift:+.1f}%")

    return X_train, y_train, X_test, y_test, feature_cols


def apply_smote(X_train, y_train, strategy='auto'):
    """Apply SMOTE to balance training data"""
    print(f"\nApplying SMOTE (strategy={strategy})...")

    # DEBUG: Check original labels
    unique_labels_before = np.unique(y_train)
    print(f"  DEBUG: Original labels: {unique_labels_before}")

    # Use SMOTE simple (not SMOTETomek to avoid -1 labels)
    smote = SMOTE(random_state=42, sampling_strategy=strategy)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    # DEBUG: Check new labels
    unique_labels_after = np.unique(y_train_balanced)
    print(f"  DEBUG: After SMOTE labels: {unique_labels_after}")

    # Print new distribution
    tp_pct_new = (y_train_balanced == 1).sum() / len(y_train_balanced) * 100

    print(f"  Original: {len(y_train)} samples, TP={(y_train==1).sum()/len(y_train)*100:.1f}%")
    print(f"  Balanced: {len(y_train_balanced)} samples, TP={tp_pct_new:.1f}%")
    print(f"  Added: {len(y_train_balanced) - len(y_train)} synthetic samples")

    return X_train_balanced, y_train_balanced


def train_phase3_model(crypto: str, X_train, y_train, X_test, y_test,
                       use_feature_selection=False, feature_cols=None):
    """Train Phase 3 model with SMOTE-balanced data"""
    print(f"\n{'='*70}")
    print(f"TRAINING PHASE 3 MODEL: {crypto.upper()}")
    print('='*70)

    # Determine if we should use feature selection
    if use_feature_selection:
        # Load selected features
        results_dir = Path(__file__).parent / 'results'
        features_file = results_dir / f'{crypto}_selected_features_top50.json'

        if features_file.exists():
            with open(features_file) as f:
                feature_data = json.load(f)
                selected_features = feature_data['selected_feature_names']

            # Get indices
            feature_indices = [feature_cols.index(f) for f in selected_features if f in feature_cols]
            X_train = X_train[:, feature_indices]
            X_test = X_test[:, feature_indices]
            print(f"\nUsing {len(feature_indices)} selected features")
        else:
            print(f"\nFeature selection file not found, using all features")
            use_feature_selection = False

    # Baseline hyperparameters (same as Phase 1)
    if crypto == 'btc':
        scale_pos_weight = 1.35 if not use_feature_selection else 0.78
    elif crypto == 'eth':
        scale_pos_weight = 0.76
    else:  # sol
        scale_pos_weight = 0.32 if not use_feature_selection else 0.35

    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 6,
        'learning_rate': 0.05,
        'n_estimators': 200,
        'gamma': 2,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42,
        'tree_method': 'hist'
    }

    print(f"\nHyperparameters:")
    for k, v in params.items():
        print(f"  {k}: {v}")

    # Train model
    print(f"\nTraining XGBoost...")
    model = XGBClassifier(**params)
    model.fit(X_train, y_train, verbose=False)

    # Evaluate
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    print(f"\nTest Performance:")
    print(f"  Accuracy: {accuracy*100:.2f}%")
    print(f"  AUC: {auc:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"  True SL: {cm[0,0]}, False TP: {cm[0,1]}")
    print(f"  False SL: {cm[1,0]}, True TP: {cm[1,1]}")

    return model, accuracy, auc


def main():
    print("=" * 70)
    print("PHASE 3: SMOTE DATA BALANCING")
    print("=" * 70)

    results = {}

    for crypto in ['btc', 'eth', 'sol']:
        print(f"\n\n{'#'*70}")
        print(f"# PROCESSING: {crypto.upper()}")
        print(f"{'#'*70}")

        # Load data
        X_train, y_train, X_test, y_test, feature_cols = load_and_prepare_data(crypto)

        # Apply SMOTE
        X_train_balanced, y_train_balanced = apply_smote(X_train, y_train)

        # Decide on feature selection
        # BTC and SOL use feature selection, ETH uses all features
        use_features = crypto in ['btc', 'sol']

        # Train model
        model, accuracy, auc = train_phase3_model(
            crypto,
            X_train_balanced,
            y_train_balanced,
            X_test,
            y_test,
            use_feature_selection=use_features,
            feature_cols=feature_cols
        )

        # Save model
        models_dir = Path(__file__).parent.parent / 'models'
        if use_features:
            model_file = models_dir / f'{crypto}_v11_phase3_smote_features.joblib'
        else:
            model_file = models_dir / f'{crypto}_v11_phase3_smote.joblib'

        joblib.dump(model, model_file)
        print(f"\nModel saved: {model_file.name}")

        # Store results
        results[crypto] = {
            'accuracy': accuracy * 100,
            'auc': auc,
            'model_file': model_file.name,
            'use_feature_selection': use_features
        }

    # Summary
    print("\n\n" + "=" * 70)
    print("PHASE 3 SUMMARY")
    print("=" * 70)

    for crypto, res in results.items():
        print(f"\n{crypto.upper()}:")
        print(f"  Model: {res['model_file']}")
        print(f"  Accuracy: {res['accuracy']:.2f}%")
        print(f"  AUC: {res['auc']:.4f}")
        print(f"  Feature Selection: {res['use_feature_selection']}")

    print("\n\nNext: Run backtesting/phase3_backtest.py to test ROI!")


if __name__ == '__main__':
    main()
