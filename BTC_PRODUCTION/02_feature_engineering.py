"""
BTC Feature Engineering Script
===============================
Generates 90 features for PyTorch model (sequences of 60 days)

Usage:
    python 02_feature_engineering.py
"""

import sys
from pathlib import Path

# Add scripts directory to path
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR / 'scripts'))

from create_90_features import create_90_features, get_feature_columns
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = BASE_DIR / 'data' / 'cache'
CRYPTO = 'BTC'


def create_features():
    """Create enhanced features for PyTorch model"""
    logger.info(f"\n{'='*70}")
    logger.info(f"CREATING {CRYPTO} FEATURES FOR PYTORCH")
    logger.info(f"{'='*70}\n")

    # Load 1d data
    df = pd.read_csv(DATA_DIR / f'{CRYPTO.lower()}_1d_data.csv')
    df['date'] = pd.to_datetime(df['date'])

    logger.info(f"Loaded {len(df)} candles")
    logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")

    # Create 90 features
    logger.info(f"\nGenerating 90 technical features...")
    df_features = create_90_features(df)

    # Get feature columns list
    feature_cols = get_feature_columns()

    # Verify all features exist
    missing = [f for f in feature_cols if f not in df_features.columns]
    if missing:
        logger.error(f"Missing features: {missing}")
        raise ValueError(f"Missing {len(missing)} features")

    logger.info(f"✓ Created {len(feature_cols)} features")

    # Save
    output_file = DATA_DIR / f'{CRYPTO.lower()}_features.csv'
    df_features.to_csv(output_file, index=False)

    logger.info(f"\n✓ Features saved to {output_file}")
    logger.info(f"  Total columns: {len(df_features.columns)}")
    logger.info(f"  Feature columns: {len(feature_cols)}")
    logger.info(f"  Total rows: {len(df_features)}")
    logger.info(f"  Sample features: {list(feature_cols[:10])}")

    return df_features


if __name__ == "__main__":
    create_features()
