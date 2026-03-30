"""
Signal Generator - CNN Inference + Filter Chain
================================================
"""

import sys
import torch
import numpy as np
import pandas as pd
import json
import joblib
import logging
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent / 'scripts'))
from direction_prediction_model import CNNDirectionModel

from config import MODEL_DIR, COINS, FILTERS, TRADING

logger = logging.getLogger(__name__)


class SignalGenerator:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_lists = {}
        self.consecutive_losses = {coin: 0 for coin in COINS}
        self.cooldown_until = {coin: None for coin in COINS}

    def load_models(self):
        """Load all CNN models, scalers, and feature lists"""
        for coin, cfg in COINS.items():
            try:
                # Load model
                model_path = MODEL_DIR / cfg['model_file']
                if not model_path.exists():
                    logger.warning(f"{coin}: Model not found at {model_path}")
                    continue

                ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
                feat_dim = ckpt.get('feature_dim', 99)
                seq_len = ckpt.get('sequence_length', 30)

                model = CNNDirectionModel(feature_dim=feat_dim, sequence_length=seq_len, dropout=0.4)
                model.load_state_dict(ckpt['model_state_dict'])
                model.eval()
                self.models[coin] = model

                # Load scaler
                scaler_path = MODEL_DIR / cfg['scaler_file']
                if scaler_path.exists():
                    self.scalers[coin] = joblib.load(scaler_path)

                # Load feature list
                feat_path = MODEL_DIR / cfg['features_file']
                if feat_path.exists():
                    with open(feat_path) as f:
                        self.feature_lists[coin] = json.load(f)

                logger.info(f"{coin}: Model loaded (feat={feat_dim}, seq={seq_len})")

            except Exception as e:
                logger.error(f"{coin}: Failed to load model: {e}")

    def predict(self, coin, scaled_features):
        """Run CNN inference. Returns (direction, confidence) or (None, None)"""
        if coin not in self.models:
            return None, None

        try:
            X = torch.tensor(scaled_features, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                direction, confidence = self.models[coin].predict_direction(X)
            return direction.item(), confidence.item()
        except Exception as e:
            logger.error(f"{coin}: Prediction error: {e}")
            return None, None

    def check_filters(self, coin, raw_row, confidence):
        """Apply intelligent filters. Returns (passes, reason)"""
        cfg = COINS[coin]
        f = FILTERS

        # 0. Confidence threshold
        if confidence < cfg['confidence_threshold']:
            return False, f"low_confidence ({confidence:.1%} < {cfg['confidence_threshold']:.1%})"

        # 1. Cooldown
        now = datetime.utcnow()
        if self.cooldown_until[coin] and now < self.cooldown_until[coin]:
            return False, f"cooldown until {self.cooldown_until[coin].strftime('%Y-%m-%d')}"

        # 2. Momentum alignment
        bullish = 0
        total_tf = 0
        for col in ['1d_momentum_5', '4h_momentum_5', '1w_momentum_5']:
            if col in raw_row.index and pd.notna(raw_row[col]):
                total_tf += 1
                if raw_row[col] > 0:
                    bullish += 1
        if total_tf > 0 and bullish < f['min_momentum_alignment']:
            return False, f"weak_momentum ({bullish}/{total_tf})"

        # 3. Volatility regime
        if 'volatility_regime' in raw_row.index and pd.notna(raw_row['volatility_regime']):
            if raw_row['volatility_regime'] > f['max_volatility_regime']:
                return False, f"high_volatility ({raw_row['volatility_regime']:.2f})"

        # 4. ADX
        if '1d_adx_14' in raw_row.index and pd.notna(raw_row['1d_adx_14']):
            if raw_row['1d_adx_14'] < f['min_adx']:
                return False, "no_trend (low ADX)"

        # 5. Bear market SMA50
        if 'distance_from_sma50' in raw_row.index and pd.notna(raw_row['distance_from_sma50']):
            if raw_row['distance_from_sma50'] < f['bear_sma50_threshold']:
                return False, f"bear_sma50 ({raw_row['distance_from_sma50']:.1%})"

        # 6. Bear market SMA20
        if 'distance_from_sma20' in raw_row.index and pd.notna(raw_row['distance_from_sma20']):
            if raw_row['distance_from_sma20'] < f['bear_sma20_threshold']:
                return False, f"bear_sma20 ({raw_row['distance_from_sma20']:.1%})"

        # 7. High vol + low confidence
        if 'volatility_regime' in raw_row.index and pd.notna(raw_row['volatility_regime']):
            if raw_row['volatility_regime'] > 1.5 and confidence < 0.65:
                return False, "low_conf_high_vol"

        # 8. RSI overbought multi-TF
        overbought = 0
        for col in ['1d_rsi_14', '4h_rsi_14', '1w_rsi_14']:
            if col in raw_row.index and pd.notna(raw_row[col]) and raw_row[col] > 70:
                overbought += 1
        if overbought >= 2:
            return False, "overbought_multi_tf"

        # 9. Trend score
        if 'trend_score' in raw_row.index and pd.notna(raw_row['trend_score']):
            if raw_row['trend_score'] < f['max_trend_score']:
                return False, f"downtrend ({raw_row['trend_score']:.0f})"

        return True, "pass"

    def get_dynamic_tp_sl(self, raw_row, entry_price):
        """Calculate TP/SL based on ATR"""
        if not TRADING['use_dynamic_tp_sl']:
            return TRADING['tp_pct'], TRADING['sl_pct']

        atr = None
        if '1d_atr_14' in raw_row.index and pd.notna(raw_row['1d_atr_14']):
            atr = raw_row['1d_atr_14']

        if atr and atr > 0:
            tp_mult = min(max(atr / entry_price, 0.008), 0.03)
            sl_mult = tp_mult * 0.5
            return tp_mult, sl_mult

        return TRADING['tp_pct'], TRADING['sl_pct']

    def record_trade_result(self, coin, exit_type):
        """Track consecutive losses for cooldown"""
        if exit_type == 'SL':
            self.consecutive_losses[coin] += 1
            if self.consecutive_losses[coin] >= FILTERS['max_consecutive_losses']:
                self.cooldown_until[coin] = datetime.utcnow() + timedelta(days=FILTERS['cooldown_days'])
                logger.warning(f"{coin}: {self.consecutive_losses[coin]} losses -> cooldown until {self.cooldown_until[coin]}")
        else:
            self.consecutive_losses[coin] = 0
