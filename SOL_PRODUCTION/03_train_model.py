"""
SOL Model Training Script
==========================
Trains XGBoost model with intelligent signal filtering

Usage:
    python 03_train_model.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import joblib
import json
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data' / 'cache'
MODEL_DIR = BASE_DIR / 'models'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

CRYPTO = 'SOL'
TRAIN_END = '2024-12-31'  # Train until end of 2024


def train_model():
    """Train XGBoost model on historical data"""
    logger.info(f"\n{'='*70}")
    logger.info(f"TRAINING {CRYPTO} XGBOOST MODEL")
    logger.info(f"{'='*70}\n")

    # Load merged data
    df = pd.read_csv(DATA_DIR / f'{CRYPTO.lower()}_multi_tf_merged.csv')
    df['date'] = pd.to_datetime(df['date'])

    logger.info(f"Total data: {len(df)} candles")
    logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")

    # Split train/test
    df_train = df[df['date'] <= TRAIN_END].copy()
    df_test = df[df['date'] > TRAIN_END].copy()

    logger.info(f"\nTrain: {len(df_train)} candles (until {TRAIN_END})")
    logger.info(f"Test: {len(df_test)} candles (Q1 2026)")

    # Prepare features - exclude target leaks and meta columns
    exclude_cols = ['date', 'timestamp', 'label_class', 'triple_barrier_label', 'label_numeric',
                    'open', 'high', 'low', 'close', 'volume',
                    'price_target_pct', 'future_price', 'future_return']
    feature_cols = [col for col in df_train.columns if col not in exclude_cols and 'future' not in col.lower() and 'target' not in col.lower()]

    X_train = df_train[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)

    # Fix labels: map -1 -> 0 for binary classification
    y_train = df_train['triple_barrier_label'].replace(-1, 0).astype(int)

    logger.info(f"\nFeatures: {len(feature_cols)}")
    logger.info(f"Train label distribution:")
    logger.info(f"  Class 0 (SL/None): {(y_train == 0).sum()}")
    logger.info(f"  Class 1 (TP): {(y_train == 1).sum()}")

    # Train XGBoost
    logger.info(f"\nTraining XGBoost model...")

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        tree_msolod='hist'
    )

    model.fit(X_train, y_train)

    # Evaluate on train set
    y_pred_train = model.predict(X_train)
    logger.info(f"\nTrain Performance:")
    logger.info(f"\n{classification_report(y_train, y_pred_train)}")

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    logger.info(f"\nTop 10 Features:")
    for idx, row in feature_importance.head(10).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")

    # Save model
    model_file = MODEL_DIR / f'{CRYPTO.lower()}_v11_top50.joblib'
    joblib.dump(model, model_file)
    logger.info(f"\n✓ Model saved to {model_file}")

    # Save feature list
    feature_file = MODEL_DIR / f'{CRYPTO.lower()}_v11_features.json'
    with open(feature_file, 'w') as f:
        json.dump(feature_cols, f)
    logger.info(f"✓ Features saved to {feature_file}")

    # Save stats
    stats = {
        'crypto': CRYPTO,
        'model_type': 'XGBoost',
        'train_end': TRAIN_END,
        'n_features': len(feature_cols),
        'train_samples': len(X_train),
        'train_tp_samples': int((y_train == 1).sum()),
        'train_sl_samples': int((y_train == 0).sum()),
        'top_features': feature_importance.head(10)[['feature', 'importance']].to_dict('records')
    }

    stats_file = MODEL_DIR / f'{CRYPTO.lower()}_v11_top50_stats.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"✓ Stats saved to {stats_file}")

    logger.info(f"\n{'='*70}")
    logger.info(f"TRAINING COMPLETE")
    logger.info(f"{'='*70}")

    return model, feature_cols


if __name__ == "__main__":
    train_model()
