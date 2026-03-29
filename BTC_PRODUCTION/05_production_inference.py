"""
BTC Production Inference Script
================================
Real-time predictions using PyTorch model

Usage:
    python 05_production_inference.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import joblib
import ccxt
import logging

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR / 'scripts'))

from direction_prediction_model import DirectionPredictionModel
from enhanced_features_fixed import EnhancedFeatureEngineering

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_DIR = BASE_DIR / 'models'
CRYPTO = 'BTC'


def fetch_latest_data():
    """Fetch latest BTC 1d data"""
    logger.info("Fetching latest BTC data from Binance...")

    exchange = ccxt.binance({'enableRateLimit': True})
    ohlcv = exchange.fetch_ohlcv(f'{CRYPTO}/USDT', '1d', limit=100)

    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')

    logger.info(f"  ✓ Fetched {len(df)} candles")
    return df


def make_prediction():
    """Make real-time prediction"""
    logger.info(f"\n{'='*70}")
    logger.info(f"{CRYPTO} PRODUCTION INFERENCE (PYTORCH)")
    logger.info(f"{'='*70}\n")

    # Load model
    model_file = MODEL_DIR / 'BTC_direction_model.pt'
    model_data = torch.load(model_file, map_location='cpu')
    model = DirectionPredictionModel(feature_dim=90, sequence_length=60)

    if isinstance(model_data, dict) and 'model_state_dict' in model_data:
        model.load_state_dict(model_data['model_state_dict'])
    else:
        model.load_state_dict(model_data)

    model.eval()
    logger.info("✓ Model loaded")

    # Fetch latest data
    df = fetch_latest_data()

    # Create features
    feature_engineer = EnhancedFeatureEngineering()
    df_features = feature_engineer.create_all_features(df)

    # Get latest sequence (last 60 days)
    sequence = df_features.tail(60)
    feature_cols = [col for col in sequence.columns if col not in ['date', 'timestamp']]

    raw_features = sequence[feature_cols].fillna(0).values

    # Load scaler if available
    scaler_path = MODEL_DIR / 'feature_scaler.joblib'
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        raw_features = np.nan_to_num(scaler.transform(raw_features), nan=0.0, posinf=0.0, neginf=0.0)
        logger.info("Feature scaler applied")

    X = torch.tensor(raw_features, dtype=torch.float32).unsqueeze(0)

    # Predict
    with torch.no_grad():
        direction, confidence = model.predict_direction(X)

    # Get current price
    current_price = df.iloc[-1]['close']
    current_time = df.iloc[-1]['date']

    logger.info(f"\n{'='*70}")
    logger.info(f"PREDICTION RESULT")
    logger.info(f"{'='*70}\n")
    logger.info(f"Time: {current_time}")
    logger.info(f"Current Price: ${current_price:.2f}")
    logger.info(f"Prediction: {'LONG' if direction.item() == 1 else 'SHORT'}")
    logger.info(f"Confidence: {confidence.item():.2%}")

    if direction.item() == 1 and confidence.item() > 0.6:
        tp_price = current_price * 1.015
        sl_price = current_price * 0.9925

        logger.info(f"\n{'='*70}")
        logger.info(f"TRADE SIGNAL")
        logger.info(f"{'='*70}\n")
        logger.info(f"✅ ENTER LONG")
        logger.info(f"   Entry: ${current_price:.2f}")
        logger.info(f"   TP (1.5%): ${tp_price:.2f}")
        logger.info(f"   SL (0.75%): ${sl_price:.2f}")
        logger.info(f"   Confidence: {confidence.item():.2%}")
    else:
        logger.info(f"\n⛔ NO TRADE")

    logger.info(f"\n{'='*70}")


if __name__ == "__main__":
    make_prediction()
