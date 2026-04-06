"""
Signal Generator - Independent LONG + SHORT CNN Models
========================================================
Each coin has a LONG model and a SHORT model that trade independently.
"""

import sys
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
import joblib
import logging
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent / 'scripts'))
from direction_prediction_model import CNNDirectionModel, DeepCNNShortModel, DeepCNNShortModelLN

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
        # Temperature scaling per coin (from training calibration)
        self.long_temps = {}
        self.short_temps = {}
        # Meta-models (XGBoost) per coin
        self.meta_long_models = {}
        self.meta_short_models = {}
        self.meta_features = {}
        # Separate cooldowns for LONG and SHORT
        self.long_consec = {coin: 0 for coin in COINS}
        self.short_consec = {coin: 0 for coin in COINS}
        self.long_cool = {coin: None for coin in COINS}
        self.short_cool = {coin: None for coin in COINS}

    def _load_one_model(self, path):
        """Load a CNN model - auto-detect DeepCNNShortModel vs CNNDirectionModel"""
        if not path.exists():
            return None, None
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        feat_dim = ckpt.get('feature_dim', 99)
        seq_len = ckpt.get('sequence_length', 30)
        model_type = ckpt.get('model_type', 'cnn')
        keys = ckpt['model_state_dict'].keys()
        is_deep_bn = any('conv3_1' in k or 'conv9_1' in k for k in keys)
        is_deep_ln = any('ln1.weight' in k or 'ln2.weight' in k for k in keys)

        if is_deep_bn:
            model = DeepCNNShortModel(feature_dim=feat_dim, sequence_length=seq_len, dropout=0.35)
        elif is_deep_ln or (model_type == 'deep_cnn_short' and not is_deep_bn):
            model = DeepCNNShortModelLN(feature_dim=feat_dim, sequence_length=seq_len, dropout=0.30)
        else:
            model = CNNDirectionModel(feature_dim=feat_dim, sequence_length=seq_len, dropout=0.4)

        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        return model, ckpt

    def load_models(self):
        """Load LONG and SHORT models + meta-models for all coins"""
        for coin, cfg in COINS.items():
            # LONG model
            try:
                result = self._load_one_model(MODEL_DIR / cfg['long_model'])
                if result and result[0]:
                    m, ckpt = result
                    self.long_models[coin] = m
                    self.long_temps[coin] = ckpt.get('temperature', 1.0)
                    s = MODEL_DIR / cfg['long_scaler']
                    if s.exists():
                        self.long_scalers[coin] = joblib.load(s)
                    f = MODEL_DIR / cfg['long_features']
                    if f.exists():
                        with open(f) as fh:
                            self.long_features[coin] = json.load(fh)
                    logger.info(f"{coin} LONG: loaded (temp={self.long_temps[coin]:.3f})")
            except Exception as e:
                logger.error(f"{coin} LONG: {e}")

            # SHORT model
            try:
                result = self._load_one_model(MODEL_DIR / cfg['short_model'])
                if result and result[0]:
                    m, ckpt = result
                    self.short_models[coin] = m
                    self.short_temps[coin] = ckpt.get('temperature', 1.0)
                    self.short_seq_lens = getattr(self, 'short_seq_lens', {})
                    self.short_seq_lens[coin] = ckpt.get('sequence_length', 30)
                    s = MODEL_DIR / cfg['short_scaler']
                    if s.exists():
                        self.short_scalers[coin] = joblib.load(s)
                    f = MODEL_DIR / cfg['short_features']
                    if f.exists():
                        with open(f) as fh:
                            self.short_features[coin] = json.load(fh)
                    logger.info(f"{coin} SHORT: loaded (temp={self.short_temps[coin]:.3f})")
            except Exception as e:
                logger.warning(f"{coin} SHORT: {e}")

            # Meta-models (XGBoost) — only for coins that have them
            try:
                coin_lower = coin.lower()
                meta_long_path = MODEL_DIR / f'{coin_lower}_meta_long.joblib'
                meta_short_path = MODEL_DIR / f'{coin_lower}_meta_short.joblib'
                meta_feat_path = MODEL_DIR / f'{coin_lower}_meta_features.json'
                if meta_long_path.exists():
                    self.meta_long_models[coin] = joblib.load(meta_long_path)
                    logger.info(f"{coin} META LONG: loaded")
                if meta_short_path.exists():
                    self.meta_short_models[coin] = joblib.load(meta_short_path)
                    logger.info(f"{coin} META SHORT: loaded")
                if meta_feat_path.exists():
                    with open(meta_feat_path) as fh:
                        self.meta_features[coin] = json.load(fh)
            except Exception as e:
                logger.warning(f"{coin} META: {e}")

    def predict_long(self, coin, scaled_features):
        """Predict LONG with temperature-calibrated confidence"""
        if coin not in self.long_models:
            return None, None, None
        try:
            X = torch.tensor(scaled_features, dtype=torch.float32).unsqueeze(0)
            temp = self.long_temps.get(coin, 1.0)
            with torch.no_grad():
                logits = self.long_models[coin](X)
                probs = F.softmax(logits / temp, dim=1).squeeze()
            conf, direction = probs.max(0)
            return direction.item(), conf.item(), probs.cpu().numpy()
        except Exception as e:
            logger.error(f"{coin} LONG predict: {e}")
            return None, None, None

    def predict_short(self, coin, scaled_features):
        """Predict SHORT with temperature-calibrated confidence"""
        if coin not in self.short_models:
            return None, None, None
        try:
            X = torch.tensor(scaled_features, dtype=torch.float32).unsqueeze(0)
            temp = self.short_temps.get(coin, 1.0)
            with torch.no_grad():
                logits = self.short_models[coin](X)
                probs = F.softmax(logits / temp, dim=1).squeeze()
            conf, direction = probs.max(0)
            return direction.item(), conf.item(), probs.cpu().numpy()
        except Exception as e:
            logger.error(f"{coin} SHORT predict: {e}")
            return None, None, None

    def build_meta_features(self, coin, raw_row, l_conf, l_dir, s_conf, s_dir, l_probs, s_probs):
        """Build meta-model input features (matches 05_train_meta_xgboost.py)"""
        if coin not in self.meta_features:
            return None

        meta_feat_cols = self.meta_features[coin]
        mf = {}
        # CNN-derived features
        mf['long_conf'] = l_conf
        mf['long_dir'] = l_dir
        mf['short_conf'] = s_conf
        mf['short_dir'] = s_dir
        mf['long_prob_spread'] = abs(float(l_probs[1]) - float(l_probs[0]))
        mf['short_prob_spread'] = abs(float(s_probs[1]) - float(s_probs[0]))
        mf['models_agree_bull'] = int(l_dir == 1 and s_dir == 0)
        mf['models_agree_bear'] = int(l_dir == 0 and s_dir == 1)
        mf['models_conflict'] = int(l_dir == 1 and s_dir == 1)
        mf['models_neutral'] = int(l_dir == 0 and s_dir == 0)
        mf['conf_diff'] = l_conf - s_conf
        mf['max_conf'] = max(l_conf, s_conf)
        mf['min_conf'] = min(l_conf, s_conf)

        # Market context features from raw_row
        for c in meta_feat_cols:
            if c not in mf:
                val = raw_row.get(c, 0) if hasattr(raw_row, 'get') else (raw_row[c] if c in raw_row.index else 0)
                mf[c] = float(val) if pd.notna(val) else 0.0

        return np.array([[mf.get(c, 0) for c in meta_feat_cols]])

    def check_meta(self, coin, direction, meta_input):
        """Check if meta-model approves the CNN signal. Returns (approved, probability)"""
        if meta_input is None:
            return True, None  # No meta = always approve

        cfg = COINS[coin]
        if direction == 'LONG' and coin in self.meta_long_models:
            prob = self.meta_long_models[coin].predict_proba(meta_input)[0][1]
            threshold = cfg.get('long_meta_conf', 0.45)
            return prob >= threshold, prob
        elif direction == 'SHORT' and coin in self.meta_short_models:
            prob = self.meta_short_models[coin].predict_proba(meta_input)[0][1]
            threshold = cfg.get('short_meta_conf', 0.50)
            return prob >= threshold, prob

        return True, None  # No meta model for this coin/direction

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

        # Per-coin SMA overrides, fallback to global FILTERS
        sma50_thresh = cfg.get('bear_sma50', f['bear_sma50_threshold'])
        sma20_thresh = cfg.get('bear_sma20', f['bear_sma20_threshold'])

        if 'distance_from_sma50' in raw_row.index and pd.notna(raw_row['distance_from_sma50']):
            if raw_row['distance_from_sma50'] < sma50_thresh:
                return False, "bear_sma50"

        if 'distance_from_sma20' in raw_row.index and pd.notna(raw_row['distance_from_sma20']):
            if raw_row['distance_from_sma20'] < sma20_thresh:
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

    def get_dynamic_tp_sl(self, raw_row, entry_price, direction='LONG', coin='BTC'):
        """Calculate TP/SL based on ATR — symmetric for V3 coins (matches training), legacy for others"""
        atr = None
        if '1d_atr_14' in raw_row.index and pd.notna(raw_row['1d_atr_14']):
            atr = raw_row['1d_atr_14']

        cfg = COINS.get(coin, {})

        # V3 coins use symmetric ATR TP/SL (matches training labels: ATR_MULT=1.5)
        if cfg.get('v3', False):
            ATR_MULT = 1.5
            if TRADING['use_dynamic_tp_sl'] and atr and atr > 0:
                tp = min(max(ATR_MULT * atr / entry_price, 0.008), 0.04)
                sl = tp  # Symmetric
            else:
                tp, sl = 0.012, 0.012  # Fallback matches training
            return tp, sl

        # Other coins: legacy asymmetric TP/SL
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
