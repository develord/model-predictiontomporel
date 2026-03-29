"""
ETH MODEL - PRODUCTION INFERENCE
=================================

Real-time inference for production trading with intelligent signal filtering.

Usage:
    python production_inference.py

Date: 2026-03-29
"""

import pandas as pd
import numpy as np
import joblib
import ccxt
from pathlib import Path
import logging
from datetime import datetime, timedelta

# Import local feature engineering modules
from enhanced_features_enriched import create_enhanced_features
from advanced_features_nontechnical import create_advanced_nontechnical_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
CRYPTO = 'ETH'
TIMEFRAME = '1d'
MIN_CONFIDENCE = 0.65
BASE_DIR = Path(__file__).parent


def load_model():
    """Load trained model and feature columns."""
    model_dir = BASE_DIR / 'models' / 'xgboost_ultimate'

    model = joblib.load(model_dir / 'model.pkl')
    feature_cols = joblib.load(model_dir / 'feature_columns.pkl')

    logger.info(f"[OK] Model loaded with {len(feature_cols)} features")
    return model, feature_cols


def fetch_latest_data(lookback_days=200):
    """Fetch latest data from Binance."""
    logger.info(f"Fetching latest {CRYPTO} data...")

    exchange = ccxt.binance({'enableRateLimit': True})
    symbol = f'{CRYPTO}/USDT'

    since = exchange.parse8601((datetime.now() - timedelta(days=lookback_days)).isoformat())
    ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME, since=since, limit=1000)

    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    logger.info(f"[OK] Fetched {len(df)} candles (latest: {df['timestamp'].iloc[-1]})")
    return df


def prepare_features(df):
    """Create features for prediction."""
    logger.info("Creating features...")

    df = create_enhanced_features(df)
    df = create_advanced_nontechnical_features(df)

    logger.info(f"[OK] Features created ({len(df.columns)} columns)")
    return df


def filter_signal(prediction_proba, features_row):
    """Apply intelligent signal filtering."""
    confidence = abs(prediction_proba - 0.5) * 2

    # Confidence filter
    if confidence < (MIN_CONFIDENCE - 0.5) * 2:
        return False, "Low confidence"

    # Volume filter
    if features_row.get('volume_relative', 1.0) < 0.7:
        return False, "Low volume"

    # Momentum filter
    signal_type = 'LONG' if prediction_proba > 0.5 else 'SHORT'
    if signal_type == 'LONG' and features_row.get('momentum_shift_bullish', 0) == 0:
        return False, "No bullish momentum"
    elif signal_type == 'SHORT' and features_row.get('momentum_shift_bearish', 0) == 0:
        return False, "No bearish momentum"

    # Volatility filter
    if features_row.get('vol_regime_high', 0) == 1:
        return False, "High volatility"

    # Market structure filter
    market_structure = features_row.get('market_structure_score', 0)
    if signal_type == 'LONG' and market_structure < -0.4:
        return False, "Bearish structure"
    elif signal_type == 'SHORT' and market_structure > 0.4:
        return False, "Bullish structure"

    return True, "All filters passed"


def get_prediction(model, feature_cols, df):
    """Get latest prediction with filtering."""
    # Use latest row
    latest = df.iloc[-1]

    # Extract features
    X = df[feature_cols].iloc[-1:].values

    # Predict
    proba = model.predict_proba(X)[0, 1]
    confidence = abs(proba - 0.5) * 2

    # Determine signal
    if proba > 0.5 + 0.01:
        signal = 'LONG'
    elif proba < 0.5 - 0.01:
        signal = 'SHORT'
    else:
        signal = 'NEUTRAL'

    # Filter signal
    features_row = df[feature_cols].iloc[-1].to_dict()
    should_trade, filter_reason = filter_signal(proba, features_row)

    return {
        'timestamp': latest['timestamp'],
        'price': latest['close'],
        'signal': signal,
        'confidence': confidence,
        'probability': proba,
        'should_trade': should_trade,
        'filter_reason': filter_reason
    }


def main():
    """Main production inference."""
    logger.info(f"\n{'='*80}")
    logger.info(f"ETH MODEL - PRODUCTION INFERENCE")
    logger.info(f"{'='*80}")
    logger.info(f"Min Confidence: {MIN_CONFIDENCE}")
    logger.info(f"Timestamp: {datetime.now()}")

    try:
        # Load model
        model, feature_cols = load_model()

        # Fetch latest data
        df = fetch_latest_data()

        # Prepare features
        df = prepare_features(df)

        # Get prediction
        prediction = get_prediction(model, feature_cols, df)

        # Display results
        logger.info(f"\n{'='*80}")
        logger.info(f"PREDICTION RESULT")
        logger.info(f"{'='*80}")
        logger.info(f"Timestamp:     {prediction['timestamp']}")
        logger.info(f"Price:         ${prediction['price']:.2f}")
        logger.info(f"Signal:        {prediction['signal']}")
        logger.info(f"Confidence:    {prediction['confidence']*100:.1f}%")
        logger.info(f"Probability:   {prediction['probability']:.4f}")
        logger.info(f"Should Trade:  {prediction['should_trade']}")
        logger.info(f"Filter Reason: {prediction['filter_reason']}")

        if prediction['should_trade']:
            logger.info(f"\n[ACTION] TRADE SIGNAL: {prediction['signal']} @ ${prediction['price']:.2f}")
        else:
            logger.info(f"\n[SKIP] Signal filtered out: {prediction['filter_reason']}")

        logger.info(f"\n{'='*80}")
        logger.info(f"INFERENCE COMPLETE")
        logger.info(f"{'='*80}")

        return prediction

    except Exception as e:
        logger.error(f"\n[ERROR] Inference failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
