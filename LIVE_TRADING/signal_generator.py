"""
Signal Generator - Independent LONG + SHORT CNN Models
========================================================
Each coin has a LONG model and a SHORT model that trade independently.
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
from direction_prediction_model import CNNDirectionModel, DeepCNNShortModel

from config import MODEL_DIR, COINS, FILTERS, TRADING

logger = logging.getLogger(__name__)


class SignalGenerator:
    def __init__(self):
        self.long_models = {}
        self.short_models = {}
        self.long_scalers = {}
        self.short_scalers = {}
        self.long_features = {}
        self.short_features = {}
        # Separate cooldowns for LONG and SHORT
        self.long_consec = {coin: 0 for coin in COINS}
        self.short_consec = {coin: 0 for coin in COINS}
        self.long_cool = {coin: None for coin in COINS}
        self.short_cool = {coin: None for coin in COINS}

    def _load_one_model(self, path):
        """Load a CNN model - auto-detect DeepCNNShortModel vs CNNDirectionModel"""
        if not path.exists():
            return None
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        feat_dim = ckpt.get('feature_dim', 99)
        seq_len = ckpt.get('sequence_length', 30)
        model_type = ckpt.get('model_type', 'cnn')

        if model_type == 'deep_cnn_short':
            model = DeepCNNShortModel(feature_dim=feat_dim, sequence_length=seq_len, dropout=0.35)
        else:
            model = CNNDirectionModel(feature_dim=feat_dim, sequence_length=seq_len, dropout=0.4)

        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        return model

    def load_models(self):
        """Load LONG and SHORT models for all coins"""
        for coin, cfg in COINS.items():
            # LONG model
            try:
                m = self._load_one_model(MODEL_DIR / cfg['long_model'])
                if m:
                    self.long_models[coin] = m
                    s = MODEL_DIR / cfg['long_scaler']
                    if s.exists():
                        self.long_scalers[coin] = joblib.load(s)
                    f = MODEL_DIR / cfg['long_features']
                    if f.exists():
                        with open(f) as fh:
                            self.long_features[coin] = json.load(fh)
                    logger.info(f"{coin} LONG: loaded")
            except Exception as e:
                logger.error(f"{coin} LONG: {e}")

            # SHORT model
            try:
                m = self._load_one_model(MODEL_DIR / cfg['short_model'])
                if m:
                    self.short_models[coin] = m
                    s = MODEL_DIR / cfg['short_scaler']
                    if s.exists():
                        self.short_scalers[coin] = joblib.load(s)
                    f = MODEL_DIR / cfg['short_features']
                    if f.exists():
                        with open(f) as fh:
                            self.short_features[coin] = json.load(fh)
                    logger.info(f"{coin} SHORT: loaded")
            except Exception as e:
                logger.warning(f"{coin} SHORT: {e}")

    def predict_long(self, coin, scaled_features):
        if coin not in self.long_models:
            return None, None
        try:
            X = torch.tensor(scaled_features, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                d, c = self.long_models[coin].predict_direction(X)
            return d.item(), c.item()
        except Exception as e:
            logger.error(f"{coin} LONG predict: {e}")
            return None, None

    def predict_short(self, coin, scaled_features):
        if coin not in self.short_models:
            return None, None
        try:
            X = torch.tensor(scaled_features, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                d, c = self.short_models[coin].predict_direction(X)
            return d.item(), c.item()
        except Exception as e:
            logger.error(f"{coin} SHORT predict: {e}")
            return None, None

    def check_long_filters(self, coin, raw_row, confidence):
        """LONG-specific filters"""
        cfg = COINS[coin]
        f = FILTERS

        if confidence < cfg['long_conf']:
            return False, f"low_conf ({confidence:.1%})"

        now = datetime.utcnow()
        if self.long_cool[coin] and now < self.long_cool[coin]:
            return False, "cooldown"

        # Momentum: at least 1 TF bullish
        bull = sum(1 for c in ['1d_momentum_5', '4h_momentum_5', '1w_momentum_5']
                   if c in raw_row.index and pd.notna(raw_row[c]) and raw_row[c] > 0)
        total = sum(1 for c in ['1d_momentum_5', '4h_momentum_5', '1w_momentum_5']
                    if c in raw_row.index and pd.notna(raw_row[c]))
        if total > 0 and bull < 1:
            return False, "weak_momentum"

        if 'distance_from_sma50' in raw_row.index and pd.notna(raw_row['distance_from_sma50']):
            if raw_row['distance_from_sma50'] < f['bear_sma50_threshold']:
                return False, "bear_sma50"

        if 'distance_from_sma20' in raw_row.index and pd.notna(raw_row['distance_from_sma20']):
            if raw_row['distance_from_sma20'] < f['bear_sma20_threshold']:
                return False, "bear_sma20"

        if 'volatility_regime' in raw_row.index and pd.notna(raw_row['volatility_regime']):
            if raw_row['volatility_regime'] > f['max_volatility_regime']:
                return False, "high_vol"

        if 'trend_score' in raw_row.index and pd.notna(raw_row['trend_score']):
            if raw_row['trend_score'] < f['max_trend_score']:
                return False, "downtrend"

        return True, "pass"

    def check_short_filters(self, coin, raw_row, confidence):
        """SHORT-specific filters (inverted logic)"""
        cfg = COINS[coin]
        f = FILTERS

        if confidence < cfg['short_conf']:
            return False, f"low_conf ({confidence:.1%})"

        now = datetime.utcnow()
        if self.short_cool[coin] and now < self.short_cool[coin]:
            return False, "cooldown"

        # Bearish momentum: at least 1 TF bearish
        bear = sum(1 for c in ['1d_momentum_5', '4h_momentum_5', '1w_momentum_5']
                   if c in raw_row.index and pd.notna(raw_row[c]) and raw_row[c] < 0)
        total = sum(1 for c in ['1d_momentum_5', '4h_momentum_5', '1w_momentum_5']
                    if c in raw_row.index and pd.notna(raw_row[c]))
        if total > 0 and bear < 1:
            return False, "weak_bear_momentum"

        # Don't short in bull market
        if 'distance_from_sma50' in raw_row.index and pd.notna(raw_row['distance_from_sma50']):
            if raw_row['distance_from_sma50'] > 0.05:
                return False, "bull_sma50"

        if 'distance_from_sma20' in raw_row.index and pd.notna(raw_row['distance_from_sma20']):
            if raw_row['distance_from_sma20'] > 0.03:
                return False, "bull_sma20"

        if 'volatility_regime' in raw_row.index and pd.notna(raw_row['volatility_regime']):
            if raw_row['volatility_regime'] > f['max_volatility_regime']:
                return False, "high_vol"

        if 'trend_score' in raw_row.index and pd.notna(raw_row['trend_score']):
            if raw_row['trend_score'] > 3:
                return False, "uptrend"

        return True, "pass"

    def get_dynamic_tp_sl(self, raw_row, entry_price, direction='LONG'):
        """Calculate TP/SL based on ATR"""
        atr = None
        if '1d_atr_14' in raw_row.index and pd.notna(raw_row['1d_atr_14']):
            atr = raw_row['1d_atr_14']

        if direction == 'LONG':
            if TRADING['use_dynamic_tp_sl'] and atr and atr > 0:
                tp = min(max(atr / entry_price, 0.008), 0.03)
                sl = tp * 0.5
            else:
                tp, sl = TRADING['tp_pct'], TRADING['sl_pct']
            return tp, sl
        else:  # SHORT
            if TRADING['use_dynamic_tp_sl'] and atr and atr > 0:
                tp = min(max(atr / entry_price, 0.01), 0.04)
                sl = tp * 0.5
            else:
                tp, sl = 0.020, 0.010
            return tp, sl

    def record_trade_result(self, coin, direction, exit_type):
        """Track consecutive losses per direction"""
        if direction == 'LONG':
            if exit_type == 'SL':
                self.long_consec[coin] += 1
                if self.long_consec[coin] >= FILTERS['max_consecutive_losses']:
                    self.long_cool[coin] = datetime.utcnow() + timedelta(days=FILTERS['cooldown_days'])
            else:
                self.long_consec[coin] = 0
        else:
            if exit_type == 'SL':
                self.short_consec[coin] += 1
                if self.short_consec[coin] >= FILTERS['max_consecutive_losses']:
                    self.short_cool[coin] = datetime.utcnow() + timedelta(days=FILTERS['cooldown_days'])
            else:
                self.short_consec[coin] = 0
