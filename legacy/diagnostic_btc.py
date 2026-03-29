"""
Diagnostic script to understand why BTC filtering produces 0 trades.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
BACKTEST_START = '2026-01-01'
BACKTEST_END = '2026-03-24'

# Load model
model_path = BASE_DIR / 'models' / 'btc_v12_honest_split.joblib'
model = joblib.load(model_path)
logger.info(f"Model loaded: {model_path}")

# Load data
data_file = BASE_DIR / 'data' / 'cache' / 'btc_multi_tf_merged.csv'
df = pd.read_csv(data_file)
df['date'] = pd.to_datetime(df['date'])

# Filter backtest period
df_test = df[(df['date'] >= BACKTEST_START) & (df['date'] <= BACKTEST_END)].copy()
logger.info(f"Test period: {len(df_test)} days")

# Identify feature columns
exclude_cols = ['date', 'open', 'high', 'low', 'close', 'volume',
               'label_class', 'label_numeric', 'price_target_pct',
               'future_price', 'triple_barrier_label']

feature_cols = [col for col in df_test.columns if col not in exclude_cols]
logger.info(f"Features: {len(feature_cols)}")

# Generate predictions (using only base 284 features, NO V12 features)
X = df_test[feature_cols].fillna(0).values
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

predictions_proba = model.predict_proba(X)[:, 1]
df_test['prediction_proba'] = predictions_proba

# Analyze predictions
logger.info(f"\n{'='*80}")
logger.info(f"PREDICTION ANALYSIS")
logger.info(f"{'='*80}")
logger.info(f"Min confidence: {predictions_proba.min():.3f}")
logger.info(f"Max confidence: {predictions_proba.max():.3f}")
logger.info(f"Mean confidence: {predictions_proba.mean():.3f}")
logger.info(f"Median confidence: {np.median(predictions_proba):.3f}")

# Count signals at different thresholds
for threshold in [0.37, 0.45, 0.50, 0.55, 0.60, 0.65]:
    count = (predictions_proba > threshold).sum()
    pct = count / len(predictions_proba) * 100
    logger.info(f"Signals > {threshold:.2f}: {count} ({pct:.1f}%)")

# Show top 10 highest confidence days
df_test_sorted = df_test.sort_values('prediction_proba', ascending=False).head(10)
logger.info(f"\nTop 10 highest confidence days:")
for idx, row in df_test_sorted.iterrows():
    logger.info(f"  {row['date'].date()}: {row['prediction_proba']:.3f} | Close: ${row['close']:.2f}")
