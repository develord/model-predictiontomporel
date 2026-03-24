"""
V12 Training - V11 Labels + ATR + LSTM Features + Dynamic TP/SL Execution
==========================================================================
Strategy:
1. Train LSTM on data < lstm_cutoff (1 year before XGBoost test)
2. Generate LSTM features (proba, confidence, signal, agrees_rsi)
3. Train XGBoost on V11 labels + all features + ATR + LSTM features
4. Dynamic ATR TP/SL applied at execution only

Anti-leakage:
- LSTM trained on data < 2024 (never sees 2024-2025)
- XGBoost trained on data < 2025 (LSTM features are out-of-sample for 2024+)
"""

import sys
import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix,
    precision_score, recall_score, f1_score
)
import xgboost as xgb
import joblib

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from v12.features.dynamic_labels import calculate_atr_series
from v12.features.lstm_features import (
    build_lstm_features_for_crypto, save_lstm_model
)

LSTM_FEATURES_COLS = ['lstm_proba', 'lstm_confidence', 'lstm_signal', 'lstm_agrees_rsi']


def load_merged_data(crypto: str) -> pd.DataFrame:
    cache_file = PROJECT_ROOT / 'data' / 'cache' / f'{crypto}_multi_tf_merged.csv'
    if not cache_file.exists():
        raise FileNotFoundError(f"Not found: {cache_file}")
    df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
    print(f"  Loaded {crypto.upper()}: {len(df)} rows, {len(df.columns)} columns")
    return df


def load_sol_top50_features() -> list:
    f = PROJECT_ROOT / 'optimization' / 'results' / 'sol_selected_features_top50.json'
    if f.exists():
        with open(f) as fh:
            data = json.load(fh)
        features = data.get('selected_feature_names', [])
        if features:
            print(f"  Loaded SOL top 50 features from V11")
            return features
    return None


def prepare_binary_target(df: pd.DataFrame, crypto: str = 'btc'):
    """
    Prepare features + target. Includes ATR + LSTM features.
    """
    exclude_cols = [
        'open', 'high', 'low', 'close', 'volume',
        'label_class', 'label_numeric',
        'price_target_pct', 'future_price',
        'triple_barrier_label'
    ]

    all_feature_cols = [col for col in df.columns if col not in exclude_cols]

    # SOL: top 50 + ATR + LSTM
    if crypto == 'sol':
        top50 = load_sol_top50_features()
        if top50:
            feature_cols = [f for f in top50 if f in all_feature_cols]
            # Add ATR + LSTM
            for extra in ['atr_pct_14'] + LSTM_FEATURES_COLS:
                if extra in all_feature_cols and extra not in feature_cols:
                    feature_cols.append(extra)
            print(f"  SOL: Using {len(feature_cols)} features (top 50 + ATR + LSTM)")
        else:
            feature_cols = all_feature_cols
    else:
        feature_cols = all_feature_cols

    # V11 fixed labels
    df_clean = df[df['triple_barrier_label'].notna()].copy()
    df_clean = df_clean[df_clean['triple_barrier_label'] != 0].copy()

    print(f"  Features: {len(feature_cols)}")
    print(f"  Samples (excl timeout): {len(df_clean)}")

    X = df_clean[feature_cols].fillna(0).values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    y = (df_clean['triple_barrier_label'].values == 1).astype(int)

    n_sl = np.sum(y == 0)
    n_tp = np.sum(y == 1)
    print(f"  Class: SL={n_sl} ({n_sl/len(y)*100:.1f}%) | TP={n_tp} ({n_tp/len(y)*100:.1f}%)")

    return X, y, feature_cols, df_clean


def train_v12_classifier(crypto: str, use_lstm: bool = True, params: dict = None):
    """Train V12: V11 labels + ATR + LSTM features."""

    print(f"\n{'='*80}")
    print(f"V12 Training: {crypto.upper()} (V11 labels + ATR + LSTM)")
    print(f"WALK-FORWARD: Train <2025, Test >=2025")
    print('='*80)

    # [1] Load data
    print(f"\n[1/5] Loading Data")
    df = load_merged_data(crypto)

    # [2] Add ATR
    print(f"\n[2/5] Adding ATR Feature")
    df['atr_pct_14'] = calculate_atr_series(df, period=14)

    # [3] LSTM features
    if use_lstm:
        print(f"\n[3/5] Training LSTM & Generating Features")
        # LSTM trained on data < 2024 (anti-leakage: 1 year gap before XGB test)
        lstm_features, lstm_model = build_lstm_features_for_crypto(
            crypto,
            lstm_train_end='2024-01-01',
            seq_len=20,
            epochs=30,
            verbose=True
        )

        # Add LSTM features to dataframe
        for col in LSTM_FEATURES_COLS:
            df[col] = lstm_features[col].values

        # Save LSTM model
        save_lstm_model(lstm_model, crypto, {
            'lstm_train_end': '2024-01-01',
            'seq_len': 20,
            'input_features': len([f for f in df.columns if f in lstm_features.columns])
        })
        print(f"  LSTM model saved")
    else:
        print(f"\n[3/5] Skipping LSTM (disabled)")

    # [4] Prepare target
    print(f"\n[4/5] Preparing XGBoost Target")
    X, y, feature_cols, df_clean = prepare_binary_target(df, crypto)

    # Temporal split
    timestamps = df_clean.index
    train_mask = timestamps < '2025-01-01'
    test_mask = timestamps >= '2025-01-01'

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    n_neg = np.sum(y_train == 0)
    n_pos = np.sum(y_train == 1)
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    print(f"\n  Train: {len(X_train)} | Test: {len(X_test)} | Weight: {scale_pos_weight:.3f}")

    # [5] Train XGBoost
    print(f"\n[5/5] Training XGBoost")
    if params is None:
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

    model = xgb.XGBClassifier(**params, verbosity=0)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    prec_tp = precision_score(y_test, y_pred, pos_label=1)
    rec_tp = recall_score(y_test, y_pred, pos_label=1)
    f1_tp = f1_score(y_test, y_pred, pos_label=1)

    print(f"\n  Results:")
    print(f"    Accuracy: {acc*100:.2f}% | AUC: {auc:.4f}")
    print(f"    TP: Prec={prec_tp:.4f} Rec={rec_tp:.4f} F1={f1_tp:.4f}")
    print(f"    CM: TN={tn} FP={fp} FN={fn} TP={tp}")
    print(f"    P(TP): [{y_proba.min():.4f}, {y_proba.max():.4f}] mean={y_proba.mean():.4f}")

    # LSTM feature importance
    if use_lstm:
        importances = model.feature_importances_
        lstm_imp = {}
        for col in LSTM_FEATURES_COLS:
            if col in feature_cols:
                idx = feature_cols.index(col)
                lstm_imp[col] = float(importances[idx])
        print(f"\n  LSTM Feature Importances:")
        for k, v in sorted(lstm_imp.items(), key=lambda x: -x[1]):
            # Rank among all features
            rank = sum(1 for x in importances if x > v) + 1
            print(f"    {k}: {v:.6f} (rank #{rank}/{len(feature_cols)})")

    # Save
    models_dir = Path(__file__).parent.parent / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)

    model_file = models_dir / f'{crypto}_v12_dynamic_atr.joblib'
    stats_file = models_dir / f'{crypto}_v12_stats.json'
    features_file = models_dir / f'{crypto}_v12_features.json'

    joblib.dump(model, model_file)
    with open(features_file, 'w') as f:
        json.dump({'feature_cols': feature_cols}, f)

    stats = {
        'crypto': crypto,
        'version': 'v12_atr_lstm',
        'use_lstm': use_lstm,
        'features': len(feature_cols),
        'lstm_features': LSTM_FEATURES_COLS if use_lstm else [],
        'train_samples': int(len(X_train)),
        'test_samples': int(len(X_test)),
        'test_metrics': {
            'accuracy': float(acc), 'auc': float(auc),
            'tp_precision': float(prec_tp), 'tp_recall': float(rec_tp), 'tp_f1': float(f1_tp)
        }
    }

    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\n  Saved: {model_file.name}")
    return model, stats, feature_cols


def train_all_cryptos():
    """Train V12 for all cryptos."""
    print("=" * 80)
    print("V12 TRAINING - V11 Labels + ATR + LSTM → XGBoost")
    print("=" * 80)

    cryptos = ['btc', 'eth', 'sol']
    results = {}

    for crypto in cryptos:
        try:
            model, stats, features = train_v12_classifier(crypto, use_lstm=True)
            results[crypto] = stats
        except Exception as e:
            print(f"\nERROR training {crypto}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print(f"\n\n{'='*80}")
    print("V12 TRAINING SUMMARY (with LSTM)")
    print('='*80)
    print(f"\n{'Crypto':<8} {'Acc':>8} {'AUC':>8} {'TP Prec':>9} {'TP Rec':>8} {'Feats':>7}")
    print("-" * 50)

    for c in cryptos:
        if c not in results:
            print(f"{c.upper():<8} FAILED")
            continue
        m = results[c]['test_metrics']
        print(f"{c.upper():<8} {m['accuracy']*100:>7.2f}% {m['auc']:>7.4f} "
              f"{m['tp_precision']:>8.4f} {m['tp_recall']:>7.4f} {results[c]['features']:>7}")

    return results


if __name__ == '__main__':
    train_all_cryptos()
