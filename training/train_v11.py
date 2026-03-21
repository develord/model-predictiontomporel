"""
V11 PRO Training - Binary Classifier for TP/SL Prediction
===========================================================
Single XGBoost binary classifier that predicts P(TP)

Key improvements over V10:
- Correct approach: Binary classification (not regression on binary labels!)
- Single model (not dual)
- Output: P(TP) probability [0-1]
- Simple and interpretable

Target: triple_barrier_label
  -1 = SL (Stop Loss hit first)
  +1 = TP (Take Profit hit first)

Converted to binary: 0 = SL, 1 = TP
"""

import sys
import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix,
    precision_score, recall_score, f1_score, classification_report
)
import xgboost as xgb
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_merged_data(crypto: str) -> pd.DataFrame:
    """Load merged multi-TF dataset from cache"""
    cache_file = Path(__file__).parent.parent / 'data' / 'cache' / f'{crypto}_multi_tf_merged.csv'

    if not cache_file.exists():
        raise FileNotFoundError(f"Merged data not found: {cache_file}")

    df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
    print(f"  Loaded {crypto.upper()}: {len(df)} rows, {len(df.columns)} columns")

    return df


def prepare_binary_target(df: pd.DataFrame):
    """
    Prepare binary classification target from triple_barrier_label

    Args:
        df: Merged dataframe with triple_barrier_label

    Returns:
        X: Features (numpy array)
        y: Binary labels (0=SL, 1=TP)
        feature_cols: List of feature column names
    """
    # Exclude non-feature columns
    exclude_cols = [
        'open', 'high', 'low', 'close', 'volume',
        'label_class', 'label_numeric',
        'price_target_pct', 'future_price',
        'triple_barrier_label'  # This is our target
    ]

    feature_cols = [col for col in df.columns if col not in exclude_cols]

    # Remove rows with NaN in triple_barrier_label
    df_clean = df[df['triple_barrier_label'].notna()].copy()

    print(f"  Features: {len(feature_cols)}")
    print(f"  Total samples: {len(df_clean)}")

    # Extract features
    X = df_clean[feature_cols].fillna(0).values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Binary target: Convert -1/+1 to 0/1
    # -1 (SL) -> 0, +1 (TP) -> 1
    y_raw = df_clean['triple_barrier_label'].values
    y = (y_raw == 1).astype(int)

    # Check distribution
    n_sl = np.sum(y == 0)
    n_tp = np.sum(y == 1)

    print(f"  Class distribution:")
    print(f"    SL (0): {n_sl} ({n_sl/len(y)*100:.1f}%)")
    print(f"    TP (1): {n_tp} ({n_tp/len(y)*100:.1f}%)")

    if n_sl == 0 or n_tp == 0:
        raise ValueError("One class is missing! Cannot train binary classifier.")

    return X, y, feature_cols


def train_binary_classifier(crypto: str, params: dict = None):
    """
    Train XGBoost binary classifier for TP/SL prediction

    Args:
        crypto: 'btc', 'eth', or 'sol'
        params: XGBoost hyperparameters (optional, uses baseline if None)

    Returns:
        model: Trained XGBClassifier
        stats: Training statistics dictionary
        feature_cols: List of feature names
    """
    print(f"\n{'='*80}")
    print(f"V11 TEMPORAL - Training Binary Classifier: {crypto.upper()}")
    print(f"WALK-FORWARD VALIDATION: Train <2025, Test >=2025")
    print('='*80)

    # Load data
    print(f"\n[1/4] Loading Data")
    df = load_merged_data(crypto)

    # Prepare binary target
    print(f"\n[2/4] Preparing Binary Target")
    X, y, feature_cols = prepare_binary_target(df)

    # TEMPORAL SPLIT: Train until 2024, Test on 2025+
    # Get timestamps from original df
    df_clean = df[df['triple_barrier_label'].notna()].copy()
    timestamps = df_clean.index

    # Split by date
    train_mask = timestamps < '2025-01-01'
    test_mask = timestamps >= '2025-01-01'

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    print(f"\n  TEMPORAL VALIDATION:")
    print(f"    Train period: {timestamps[train_mask][0]} to {timestamps[train_mask][-1]}")
    print(f"    Test period:  {timestamps[test_mask][0]} to {timestamps[test_mask][-1]}")

    print(f"\n[3/4] Training Binary Classifier")
    print(f"  Train samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")

    # Calculate class weight for imbalance
    n_neg = np.sum(y_train == 0)
    n_pos = np.sum(y_train == 1)
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    print(f"  Class imbalance weight: {scale_pos_weight:.2f}")

    # Default baseline params (better than V10's conservative ones)
    if params is None:
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 6,                  # Deeper than V10's 3
            'learning_rate': 0.05,           # Higher than V10's 0.01
            'n_estimators': 200,
            'gamma': 2,                      # Less conservative than V10's 5
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'scale_pos_weight': scale_pos_weight,
            'random_state': 42,
            'tree_method': 'hist'
        }

    print(f"  Hyperparameters:")
    print(f"    max_depth: {params['max_depth']}")
    print(f"    learning_rate: {params['learning_rate']}")
    print(f"    n_estimators: {params['n_estimators']}")
    print(f"    gamma: {params['gamma']}")

    # Train model
    model = xgb.XGBClassifier(**params, verbosity=1)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    # Evaluate
    print(f"\n[4/4] Evaluating Model")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # P(TP)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # Precision, Recall, F1 for TP class
    precision_tp = precision_score(y_test, y_pred, pos_label=1)
    recall_tp = recall_score(y_test, y_pred, pos_label=1)
    f1_tp = f1_score(y_test, y_pred, pos_label=1)

    # Precision, Recall, F1 for SL class
    precision_sl = precision_score(y_test, y_pred, pos_label=0)
    recall_sl = recall_score(y_test, y_pred, pos_label=0)
    f1_sl = f1_score(y_test, y_pred, pos_label=0)

    print(f"\n  Test Results:")
    print(f"    Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"    AUC: {auc:.4f}")
    print(f"\n  TP (Take Profit) Class:")
    print(f"    Precision: {precision_tp:.4f}")
    print(f"    Recall: {recall_tp:.4f}")
    print(f"    F1-Score: {f1_tp:.4f}")
    print(f"\n  SL (Stop Loss) Class:")
    print(f"    Precision: {precision_sl:.4f}")
    print(f"    Recall: {recall_sl:.4f}")
    print(f"    F1-Score: {f1_sl:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    True SL (TN):  {tn:5d}")
    print(f"    False TP (FP): {fp:5d}")
    print(f"    False SL (FN): {fn:5d}")
    print(f"    True TP (TP):  {tp:5d}")

    # Probability distribution analysis
    print(f"\n  Probability Distribution:")
    print(f"    P(TP) range: {y_pred_proba.min():.4f} to {y_pred_proba.max():.4f}")
    print(f"    P(TP) mean: {y_pred_proba.mean():.4f}")
    print(f"    P(TP) std: {y_pred_proba.std():.4f}")

    # Stats dictionary
    stats = {
        'crypto': crypto,
        'version': 'v11_pro',
        'features': len(feature_cols),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'class_distribution_train': {
            'SL': int(n_neg),
            'TP': int(n_pos),
            'SL_pct': float(n_neg / len(y_train) * 100),
            'TP_pct': float(n_pos / len(y_train) * 100)
        },
        'test_metrics': {
            'accuracy': float(accuracy),
            'auc': float(auc),
            'tp_class': {
                'precision': float(precision_tp),
                'recall': float(recall_tp),
                'f1': float(f1_tp)
            },
            'sl_class': {
                'precision': float(precision_sl),
                'recall': float(recall_sl),
                'f1': float(f1_sl)
            },
            'confusion_matrix': {
                'true_sl': int(tn),
                'false_tp': int(fp),
                'false_sl': int(fn),
                'true_tp': int(tp)
            },
            'prob_distribution': {
                'min': float(y_pred_proba.min()),
                'max': float(y_pred_proba.max()),
                'mean': float(y_pred_proba.mean()),
                'std': float(y_pred_proba.std())
            }
        },
        'hyperparameters': params
    }

    # Save model and stats
    models_dir = Path(__file__).parent.parent / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)

    model_file = models_dir / f'{crypto}_v11_classifier.joblib'
    stats_file = models_dir / f'{crypto}_v11_stats.json'

    joblib.dump(model, model_file)

    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\n  Saved:")
    print(f"    Model: {model_file}")
    print(f"    Stats: {stats_file}")

    return model, stats, feature_cols


def train_all_cryptos():
    """Train V11 binary classifier for all cryptos"""

    print("="*80)
    print("V11 PRO - BINARY CLASSIFIER TRAINING")
    print("="*80)
    print("\nApproach: Single XGBoost binary classifier")
    print("Target: P(TP) - Probability of hitting Take Profit")
    print("Output: [0-1] probability for trading decision")

    cryptos_list = ['btc', 'eth', 'sol']
    results = {}

    for crypto in cryptos_list:
        try:
            model, stats, feature_cols = train_binary_classifier(crypto)
            results[crypto] = stats
        except Exception as e:
            print(f"\nERROR training {crypto}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Summary
    print(f"\n\n{'='*80}")
    print("TRAINING SUMMARY")
    print('='*80)

    for crypto in cryptos_list:
        if crypto not in results:
            print(f"\n{crypto.upper()}: FAILED")
            continue

        stats = results[crypto]
        print(f"\n{crypto.upper()}:")
        print(f"  Features: {stats['features']}")
        print(f"  Test Accuracy: {stats['test_metrics']['accuracy']:.4f} ({stats['test_metrics']['accuracy']*100:.2f}%)")
        print(f"  AUC: {stats['test_metrics']['auc']:.4f}")
        print(f"  TP Precision: {stats['test_metrics']['tp_class']['precision']:.4f}")
        print(f"  TP Recall: {stats['test_metrics']['tp_class']['recall']:.4f}")
        print(f"  TP F1: {stats['test_metrics']['tp_class']['f1']:.4f}")

    # Compare to V10
    print(f"\n\n{'='*80}")
    print("V11 vs V10 COMPARISON")
    print('='*80)
    print("\nV10 (FAILED):")
    print("  BTC: 30.83% accuracy (worse than random)")
    print("  ETH: 39.17% accuracy")
    print("  SOL: 36.43% accuracy")
    print("  Regression R2: ~0% (useless)")
    print("\nV11 (NEW):")
    for crypto in cryptos_list:
        if crypto in results:
            acc = results[crypto]['test_metrics']['accuracy']
            auc = results[crypto]['test_metrics']['auc']
            improvement = "MUCH BETTER!" if acc > 0.55 else "Still needs work"
            print(f"  {crypto.upper()}: {acc*100:.2f}% accuracy, AUC={auc:.4f} - {improvement}")

    print(f"\n{'='*80}")
    print("V11 BINARY CLASSIFIER TRAINING COMPLETE!")
    print('='*80)

    return results


if __name__ == '__main__':
    results = train_all_cryptos()
