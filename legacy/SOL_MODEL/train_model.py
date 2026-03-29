"""
SOL MODEL - COMPLETE TRAINING PIPELINE
======================================

Complete independent training pipeline for SOL:
1. Download historical data from Binance
2. Advanced feature engineering (129 features)
3. Train XGBoost Ultimate model
4. Save model and feature importance

Date: 2026-03-29
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
import joblib
import logging
from datetime import datetime
import ccxt

# Import local feature engineering modules
from enhanced_features_enriched import create_enhanced_features
from advanced_features_nontechnical import create_advanced_nontechnical_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
CRYPTO = 'SOL'
TIMEFRAME = '1d'
SINCE_DATE = '2020-08-01'  # SOL launched in Aug 2020
BASE_DIR = Path(__file__).parent


def download_data():
    """
    Download historical SOL data from Binance.
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"STEP 1: DOWNLOADING {CRYPTO} DATA")
    logger.info(f"{'='*80}")

    data_dir = BASE_DIR / 'data'
    data_dir.mkdir(exist_ok=True, parents=True)

    output_file = data_dir / f'{CRYPTO}_{TIMEFRAME}.csv'

    # Check if data already exists
    if output_file.exists():
        logger.info(f"[OK] Data already exists: {output_file}")
        df = pd.read_csv(output_file)
        logger.info(f"[OK] Loaded {len(df)} rows from {df['timestamp'].min()} to {df['timestamp'].max()}")
        return df

    # Download from Binance
    logger.info(f"Downloading {CRYPTO}/{TIMEFRAME} from Binance since {SINCE_DATE}...")

    try:
        exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })

        symbol = f'{CRYPTO}/USDT'
        since = exchange.parse8601(f'{SINCE_DATE}T00:00:00Z')

        all_ohlcv = []

        while True:
            ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME, since=since, limit=1000)

            if not ohlcv:
                break

            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1

            logger.info(f"  Downloaded {len(all_ohlcv)} candles...")

            if len(ohlcv) < 1000:
                break

        # Convert to DataFrame
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        # Save
        df.to_csv(output_file, index=False)
        logger.info(f"[OK] Saved {len(df)} rows to {output_file}")

        return df

    except Exception as e:
        logger.error(f"[ERROR] Failed to download data: {e}")
        raise


def create_features(df):
    """
    Create advanced features (129 features total).
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"STEP 2: FEATURE ENGINEERING")
    logger.info(f"{'='*80}")

    initial_cols = len(df.columns)

    # Enhanced technical features
    logger.info("Creating enhanced technical features...")
    df = create_enhanced_features(df)

    # Advanced non-technical features
    logger.info("Creating advanced non-technical features...")
    df = create_advanced_nontechnical_features(df)

    final_cols = len(df.columns)
    new_features = final_cols - initial_cols

    logger.info(f"[OK] Created {new_features} features (Total: {final_cols} columns)")

    return df


def create_labels(df, threshold=0.01):
    """
    Create LONG vs SHORT labels with 1.0% threshold.

    Label = 1 (LONG) if future_return > +1.0%
    Label = 0 (SHORT) if future_return < -1.0%
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"STEP 3: CREATING LABELS (Threshold: {threshold*100:.1f}%)")
    logger.info(f"{'='*80}")

    # Calculate future 7-day return
    df['future_return'] = df['close'].pct_change(7).shift(-7)

    # Create labels
    df['label'] = np.where(
        df['future_return'] > threshold, 1,
        np.where(df['future_return'] < -threshold, 0, np.nan)
    )

    # Remove neutral returns
    df_labeled = df.dropna(subset=['label'])

    # Stats
    long_count = (df_labeled['label'] == 1).sum()
    short_count = (df_labeled['label'] == 0).sum()
    total = len(df_labeled)

    logger.info(f"[OK] Labels created:")
    logger.info(f"  LONG (1):  {long_count:4d} ({long_count/total*100:.1f}%)")
    logger.info(f"  SHORT (0): {short_count:4d} ({short_count/total*100:.1f}%)")
    logger.info(f"  Total:     {total:4d}")

    return df_labeled


def train_model(df):
    """
    Train XGBoost Ultimate model.
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"STEP 4: TRAINING XGBOOST ULTIMATE MODEL")
    logger.info(f"{'='*80}")

    # Prepare features
    exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'future_return', 'label']
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # Remove NaN
    df_clean = df.dropna(subset=feature_cols + ['label'])

    X = df_clean[feature_cols].values
    y = df_clean['label'].values

    # Train/Test split (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    logger.info(f"Train set: {len(X_train)} samples")
    logger.info(f"Test set:  {len(X_test)} samples")
    logger.info(f"Features:  {len(feature_cols)}")

    # XGBoost parameters
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 6,
        'learning_rate': 0.05,
        'n_estimators': 200,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1,
        'tree_method': 'hist'
    }

    logger.info(f"\nTraining XGBoost with {params['n_estimators']} trees...")

    model = xgb.XGBClassifier(**params)

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    # Evaluate
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)

    logger.info(f"\n[OK] Training complete!")
    logger.info(f"  Train Accuracy: {train_acc*100:.2f}%")
    logger.info(f"  Test Accuracy:  {test_acc*100:.2f}%")

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    logger.info(f"\nTop 10 Most Important Features:")
    for idx, row in feature_importance.head(10).iterrows():
        logger.info(f"  {row['feature']:40s}: {row['importance']:.4f}")

    return model, feature_cols, feature_importance


def save_model(model, feature_cols, feature_importance):
    """
    Save trained model and artifacts.
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"STEP 5: SAVING MODEL")
    logger.info(f"{'='*80}")

    model_dir = BASE_DIR / 'models' / 'xgboost_ultimate'
    model_dir.mkdir(exist_ok=True, parents=True)

    # Save model
    model_path = model_dir / 'model.pkl'
    joblib.dump(model, model_path)
    logger.info(f"[OK] Model saved: {model_path}")

    # Save feature columns
    feature_path = model_dir / 'feature_columns.pkl'
    joblib.dump(feature_cols, feature_path)
    logger.info(f"[OK] Feature columns saved: {feature_path}")

    # Save feature importance
    importance_path = model_dir / 'feature_importance.csv'
    feature_importance.to_csv(importance_path, index=False)
    logger.info(f"[OK] Feature importance saved: {importance_path}")

    # Save training metadata
    metadata = {
        'crypto': CRYPTO,
        'timeframe': TIMEFRAME,
        'trained_at': datetime.now().isoformat(),
        'n_features': len(feature_cols),
        'model_type': 'XGBoost Ultimate'
    }

    metadata_path = model_dir / 'metadata.pkl'
    joblib.dump(metadata, metadata_path)
    logger.info(f"[OK] Metadata saved: {metadata_path}")


def main():
    """
    Main training pipeline.
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"SOL MODEL - COMPLETE TRAINING PIPELINE")
    logger.info(f"{'='*80}")
    logger.info(f"Crypto:    {CRYPTO}")
    logger.info(f"Timeframe: {TIMEFRAME}")
    logger.info(f"Since:     {SINCE_DATE}")

    try:
        # Step 1: Download data
        df = download_data()

        # Step 2: Feature engineering
        df = create_features(df)

        # Step 3: Create labels
        df = create_labels(df, threshold=0.01)

        # Step 4: Train model
        model, feature_cols, feature_importance = train_model(df)

        # Step 5: Save model
        save_model(model, feature_cols, feature_importance)

        logger.info(f"\n{'='*80}")
        logger.info(f"SUCCESS! SOL MODEL TRAINING COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"\nNext steps:")
        logger.info(f"1. Run backtest: python backtest.py")
        logger.info(f"2. Use in production: python production_inference.py")

    except Exception as e:
        logger.error(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
