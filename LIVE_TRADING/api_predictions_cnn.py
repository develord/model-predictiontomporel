"""
CNN + Meta-XGBoost PREDICTION SERVICE (V3)
==========================================
Prediction service using CNN LONG+SHORT + XGBoost Meta models.

BTC uses:
- Temperature-calibrated confidence
- XGBoost meta-model filtering
- Market regime features (28 extra)
- Bear features for SHORT (15 extra)
- ATR symmetric TP/SL (matches training)
- LONG CNN >= 75% (bear market strict)

Other coins: legacy CNN LONG+SHORT (unchanged).
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
import joblib
import ccxt
import ta
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime

from direction_prediction_model import CNNDirectionModel, DeepCNNShortModel

logger = logging.getLogger(__name__)

COIN_CONFIG = {
    'bitcoin':    {'symbol': 'BTC/USDT', 'short_name': 'btc',  'long_conf': 0.75, 'short_conf': 0.55, 'long_meta_conf': 0.45, 'short_meta_conf': 0.50, 'start': '2017-01-01'},
    'ethereum':   {'symbol': 'ETH/USDT', 'short_name': 'eth',  'long_conf': 0.60, 'short_conf': 0.55, 'start': '2018-01-01'},
    'solana':     {'symbol': 'SOL/USDT', 'short_name': 'sol',  'long_conf': 0.60, 'short_conf': 0.55, 'start': '2020-08-01'},
    'avalanche':  {'symbol': 'AVAX/USDT','short_name': 'avax', 'long_conf': 0.60, 'short_conf': 0.55, 'start': '2020-09-01'},
    'xrp':        {'symbol': 'XRP/USDT', 'short_name': 'xrp',  'long_conf': 0.55, 'short_conf': 0.68, 'start': '2018-01-01'},
    'chainlink':  {'symbol': 'LINK/USDT','short_name': 'link', 'long_conf': 0.85, 'short_conf': 0.55, 'start': '2017-12-01'},
    'near':       {'symbol': 'NEAR/USDT','short_name': 'near', 'long_conf': 0.70, 'short_conf': 0.52, 'start': '2020-10-01'},
    'filecoin':   {'symbol': 'FIL/USDT', 'short_name': 'fil',  'long_conf': 0.60, 'short_conf': 0.55, 'long_meta_conf': 0.0, 'short_meta_conf': 0.0, 'start': '2020-10-15'},
    'polygon':    {'symbol': 'MATIC/USDT','short_name': 'matic', 'long_conf': 0.60, 'short_conf': 0.55, 'long_meta_conf': 0.50, 'short_meta_conf': 0.50, 'start': '2021-05-01'},
}

SEQ_LEN = 30


class CNNPredictionService:
    def __init__(self):
        d1 = Path(__file__).parent.parent / 'models' / 'cnn'
        d2 = Path(__file__).parent / 'models' / 'cnn'
        self.models_dir = d1 if d1.exists() else d2
        self.long_models = {}
        self.short_models = {}
        self.long_scalers = {}
        self.short_scalers = {}
        self.long_features = {}
        self.short_features = {}
        self.long_seq_lens = {}
        self.short_seq_lens = {}
        self.long_temps = {}
        self.short_temps = {}
        # Meta-models (XGBoost)
        self.meta_long_models = {}
        self.meta_short_models = {}
        self.meta_features = {}
        self.exchange = ccxt.binance({'enableRateLimit': True})
        logger.info("[CNN+Meta] Prediction Service initialized")

    def _load_model(self, path, is_short=False):
        if not path.exists():
            return None, 30, 1.0
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        feature_dim = ckpt.get('feature_dim', 99)
        seq_len = ckpt.get('sequence_length', 30)
        temperature = ckpt.get('temperature', 1.0)
        model_type = ckpt.get('model_type', 'cnn')

        if model_type == 'deep_cnn_short' and is_short:
            try:
                model = DeepCNNShortModel(feature_dim=feature_dim, sequence_length=seq_len, dropout=0.35)
                model.load_state_dict(ckpt['model_state_dict'])
            except Exception:
                model = CNNDirectionModel(feature_dim=feature_dim, sequence_length=seq_len, dropout=0.4)
                model.load_state_dict(ckpt['model_state_dict'])
        else:
            model = CNNDirectionModel(feature_dim=feature_dim, sequence_length=seq_len, dropout=0.4)
            model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        return model, seq_len, temperature

    async def load_models(self):
        logger.info("[CNN+Meta] Loading models...")
        for crypto_id, cfg in COIN_CONFIG.items():
            sn = cfg['short_name']

            # LONG
            m, sl, temp = self._load_model(self.models_dir / f'{sn}_cnn_model.pt')
            if m:
                self.long_models[crypto_id] = m
                self.long_seq_lens[crypto_id] = sl
                self.long_temps[crypto_id] = temp
                s = self.models_dir / f'{sn}_feature_scaler.joblib'
                if s.exists(): self.long_scalers[crypto_id] = joblib.load(s)
                f = self.models_dir / f'{sn}_features.json'
                if f.exists():
                    with open(f) as fh: self.long_features[crypto_id] = json.load(fh)
                logger.info(f"  [OK] {crypto_id} LONG (temp={temp:.3f})")

            # SHORT
            m, sl, temp = self._load_model(self.models_dir / f'{sn}_short_cnn_model.pt', is_short=True)
            if m:
                self.short_models[crypto_id] = m
                self.short_seq_lens[crypto_id] = sl
                self.short_temps[crypto_id] = temp
                s = self.models_dir / f'{sn}_short_feature_scaler.joblib'
                if s.exists(): self.short_scalers[crypto_id] = joblib.load(s)
                f = self.models_dir / f'{sn}_short_features.json'
                if f.exists():
                    with open(f) as fh: self.short_features[crypto_id] = json.load(fh)
                logger.info(f"  [OK] {crypto_id} SHORT (temp={temp:.3f})")

            # META (XGBoost) - only for coins that have them
            ml_path = self.models_dir / f'{sn}_meta_long.joblib'
            ms_path = self.models_dir / f'{sn}_meta_short.joblib'
            mf_path = self.models_dir / f'{sn}_meta_features.json'
            if ml_path.exists():
                self.meta_long_models[crypto_id] = joblib.load(ml_path)
                logger.info(f"  [OK] {crypto_id} META LONG")
            if ms_path.exists():
                self.meta_short_models[crypto_id] = joblib.load(ms_path)
                logger.info(f"  [OK] {crypto_id} META SHORT")
            if mf_path.exists():
                with open(mf_path) as fh:
                    self.meta_features[crypto_id] = json.load(fh)

        total = len(self.long_models) + len(self.short_models)
        meta_total = len(self.meta_long_models) + len(self.meta_short_models)
        logger.info(f"[CNN+Meta] Loaded {total} CNN + {meta_total} Meta models")

    def get_live_price(self, crypto_id: str) -> Optional[float]:
        try:
            symbol = COIN_CONFIG[crypto_id]['symbol']
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker['last']
        except Exception as e:
            logger.error(f"Price error {crypto_id}: {e}")
            return None

    def _download_ohlcv(self, crypto_id: str, timeframe: str, limit: int = 300):
        symbol = COIN_CONFIG[crypto_id]['symbol']
        ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df

    def _create_indicators(self, df, prefix=''):
        df[f'{prefix}rsi_14'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        df[f'{prefix}rsi_21'] = ta.momentum.RSIIndicator(df['close'], window=21).rsi()
        macd = ta.trend.MACD(df['close'])
        df[f'{prefix}macd_line'] = macd.macd()
        df[f'{prefix}macd_signal'] = macd.macd_signal()
        df[f'{prefix}macd_histogram'] = macd.macd_diff()
        bb = ta.volatility.BollingerBands(df['close'])
        df[f'{prefix}bb_upper'] = bb.bollinger_hband()
        df[f'{prefix}bb_middle'] = bb.bollinger_mavg()
        df[f'{prefix}bb_lower'] = bb.bollinger_lband()
        df[f'{prefix}bb_width'] = bb.bollinger_wband()
        df[f'{prefix}ema_12'] = ta.trend.EMAIndicator(df['close'], window=12).ema_indicator()
        df[f'{prefix}ema_26'] = ta.trend.EMAIndicator(df['close'], window=26).ema_indicator()
        df[f'{prefix}ema_50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
        df[f'{prefix}atr_14'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
        df[f'{prefix}stoch_k'] = stoch.stoch()
        df[f'{prefix}stoch_d'] = stoch.stoch_signal()
        df[f'{prefix}adx_14'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()
        df[f'{prefix}obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
        df[f'{prefix}cmf_20'] = ta.volume.ChaikinMoneyFlowIndicator(df['high'], df['low'], df['close'], df['volume'], window=20).chaikin_money_flow()
        df[f'{prefix}momentum_5'] = df['close'].pct_change(5)
        df[f'{prefix}momentum_10'] = df['close'].pct_change(10)
        df[f'{prefix}hist_vol_20'] = df['close'].pct_change().rolling(20).std()
        df[f'{prefix}volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        return df

    def _add_cross_tf_and_non_tech(self, df):
        for tf1, tf2 in [('1d', '4h'), ('1d', '1w'), ('4h', '1w')]:
            c1, c2 = f'{tf1}_rsi_14', f'{tf2}_rsi_14'
            if c1 in df.columns and c2 in df.columns:
                df[f'rsi_diff_{tf1}_{tf2}'] = df[c1] - df[c2]
        rsi_cols = [c for c in df.columns if c.endswith('_rsi_14')]
        if len(rsi_cols) >= 2:
            df['rsi_bullish_count'] = sum((df[c] > 50).astype(int) for c in rsi_cols)
            df['rsi_oversold_count'] = sum((df[c] < 30).astype(int) for c in rsi_cols)
            df['rsi_overbought_count'] = sum((df[c] > 70).astype(int) for c in rsi_cols)
        macd_cols = [c for c in df.columns if c.endswith('_macd_histogram')]
        if len(macd_cols) >= 2:
            df['macd_bullish_count'] = sum((df[c] > 0).astype(int) for c in macd_cols)
        mom_cols = [c for c in df.columns if c.endswith('_momentum_5')]
        if len(mom_cols) >= 2:
            df['momentum_bullish_count'] = sum((df[c] > 0).astype(int) for c in mom_cols)
        adx_cols = [c for c in df.columns if c.endswith('_adx_14')]
        if len(adx_cols) >= 2:
            df['adx_strong_count'] = sum((df[c] > 25).astype(int) for c in adx_cols)
            df['adx_mean'] = sum(df[c] for c in adx_cols) / len(adx_cols)
        vol_cols = [c for c in df.columns if c.endswith('_hist_vol_20')]
        if len(vol_cols) >= 2:
            df['vol_mean_all_tf'] = sum(df[c] for c in vol_cols) / len(vol_cols)

        # Non-technical
        df['daily_range_pct'] = (df['high'] - df['low']) / df['close']
        df['daily_range_ma5'] = df['daily_range_pct'].rolling(5).mean()
        df['daily_range_ma20'] = df['daily_range_pct'].rolling(20).mean()
        df['volatility_regime'] = (df['daily_range_ma5'] / df['daily_range_ma20']).fillna(1)
        df['volume_ma5'] = df['volume'].rolling(5).mean()
        df['volume_ma20'] = df['volume'].rolling(20).mean()
        df['volume_trend'] = (df['volume_ma5'] / df['volume_ma20']).fillna(1)
        df['price_position_20'] = (df['close'] - df['low'].rolling(20).min()) / (df['high'].rolling(20).max() - df['low'].rolling(20).min() + 1e-10)
        df['price_position_50'] = (df['close'] - df['low'].rolling(50).min()) / (df['high'].rolling(50).max() - df['low'].rolling(50).min() + 1e-10)
        body = abs(df['close'] - df['open'])
        wick = df['high'] - df['low']
        df['body_ratio'] = body / (wick + 1e-10)
        df['upper_shadow_ratio'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (wick + 1e-10)
        df['lower_shadow_ratio'] = (df[['open', 'close']].min(axis=1) - df['low']) / (wick + 1e-10)
        df['returns'] = df['close'].pct_change()
        df['consecutive_up'] = 0
        df['consecutive_down'] = 0
        for i in range(1, len(df)):
            if df.iloc[i]['returns'] > 0:
                df.iloc[i, df.columns.get_loc('consecutive_up')] = df.iloc[i-1]['consecutive_up'] + 1
            if df.iloc[i]['returns'] < 0:
                df.iloc[i, df.columns.get_loc('consecutive_down')] = df.iloc[i-1]['consecutive_down'] + 1
        df['distance_from_sma20'] = (df['close'] / df['close'].rolling(20).mean() - 1)
        df['distance_from_sma50'] = (df['close'] / df['close'].rolling(50).mean() - 1)
        df['higher_highs'] = (df['high'] > df['high'].shift(1)).rolling(5).sum()
        df['lower_lows'] = (df['low'] < df['low'].shift(1)).rolling(5).sum()
        df['trend_score'] = df['higher_highs'] - df['lower_lows']
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_month'] = df['date'].dt.day
        df['month'] = df['date'].dt.month
        return df

    def _add_market_regime_features(self, df):
        """Market regime features — matches BTC training exactly"""
        sma_50 = df['close'].rolling(50).mean()
        sma_200 = df['close'].rolling(200).mean()
        df['sma_50'] = sma_50
        df['sma_200'] = sma_200
        df['sma50_above_sma200'] = (sma_50 > sma_200).astype(int)
        df['sma_spread_pct'] = (sma_50 / sma_200 - 1) * 100
        ema_20 = df['close'].rolling(20).mean()
        slope_20 = ema_20.pct_change(5) * 100
        df['regime_slope'] = slope_20
        df['regime_bull'] = ((slope_20 > 0.5) & (df['close'] > sma_50)).astype(int)
        df['regime_bear'] = ((slope_20 < -0.5) & (df['close'] < sma_50)).astype(int)
        df['regime_range'] = ((slope_20.abs() <= 0.5)).astype(int)
        high_20 = df['high'].rolling(20).max()
        low_20 = df['low'].rolling(20).min()
        df['resistance_dist_pct'] = (high_20 / df['close'] - 1) * 100
        df['support_dist_pct'] = (1 - low_20 / df['close']) * 100
        df['sr_range_pct'] = (high_20 - low_20) / df['close'] * 100
        high_50 = df['high'].rolling(50).max()
        low_50 = df['low'].rolling(50).min()
        df['resistance_50_dist_pct'] = (high_50 / df['close'] - 1) * 100
        df['support_50_dist_pct'] = (1 - low_50 / df['close']) * 100
        vol_trend = df['volume'].rolling(10).mean() / df['volume'].rolling(30).mean()
        price_pos = (df['close'] - low_20) / (high_20 - low_20 + 1e-10)
        df['volume_trend_ratio'] = vol_trend
        df['price_position_in_range'] = price_pos
        df['accumulation_score'] = ((df['regime_range'] == 1).astype(float) * (vol_trend > 1).astype(float) * (price_pos < 0.3).astype(float))
        df['distribution_score'] = ((df['regime_range'] == 1).astype(float) * (vol_trend > 1).astype(float) * (price_pos > 0.7).astype(float))
        sma_cross = (sma_50 > sma_200).astype(int).diff()
        df['golden_cross_5d'] = sma_cross.rolling(5).sum().clip(0, 1)
        df['death_cross_5d'] = (-sma_cross).rolling(5).sum().clip(0, 1)
        rsi = df['1d_rsi_14'] if '1d_rsi_14' in df.columns else ta.momentum.RSIIndicator(df['close'], 14).rsi()
        df['rsi_regime_shift'] = rsi.diff(5)
        returns = df['close'].pct_change()
        vol_weighted_return = (returns * df['volume']).rolling(10).sum() / df['volume'].rolling(10).sum()
        df['vwap_trend_10'] = vol_weighted_return * 100
        df['buying_pressure'] = ((df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10))
        df['selling_pressure'] = ((df['high'] - df['close']) / (df['high'] - df['low'] + 1e-10))
        df['pressure_ratio'] = df['buying_pressure'] / (df['selling_pressure'] + 1e-10)
        df['trend_consistency_10'] = returns.rolling(10).apply(lambda x: (x > 0).sum() / len(x), raw=True)
        df['trend_consistency_20'] = returns.rolling(20).apply(lambda x: (x > 0).sum() / len(x), raw=True)
        sma50_val = df['close'].rolling(50).mean()
        sma_dist = df['close'] / sma50_val - 1
        rsi_val = df['1d_rsi_14'] if '1d_rsi_14' in df.columns else rsi
        df['is_strong_bear'] = ((sma_dist < -0.10) | ((rsi_val < 30) & (sma_dist < -0.05))).astype(int)
        return df

    def _add_bear_features(self, df):
        """Bear features for SHORT model — matches BTC training exactly"""
        for w in [10, 20, 50]:
            sma = df['close'].rolling(w).mean()
            df[f'price_above_sma{w}_pct'] = (df['close'] / sma - 1) * 100
        df['roc_5'] = df['close'].pct_change(5) * 100
        df['roc_10'] = df['close'].pct_change(10) * 100
        df['roc_deceleration'] = df['roc_5'] - df['roc_10']
        price_change_5 = df['close'].pct_change(5)
        vol_change_5 = df['volume'].pct_change(5)
        df['vol_price_divergence'] = np.where((price_change_5 > 0) & (vol_change_5 < 0), 1, np.where((price_change_5 < 0) & (vol_change_5 > 0), -1, 0))
        df['is_red'] = (df['close'] < df['open']).astype(int)
        df['consec_red'] = 0
        for i in range(1, len(df)):
            if df.iloc[i]['is_red']:
                df.iloc[i, df.columns.get_loc('consec_red')] = df.iloc[i-1]['consec_red'] + 1
        df['high_rejection'] = (df['high'] - df['close']) / (df['high'] - df['low'] + 1e-10)
        df['dist_from_high_20'] = (df['close'] / df['high'].rolling(20).max() - 1) * 100
        df['dist_from_high_50'] = (df['close'] / df['high'].rolling(50).max() - 1) * 100
        df['vol_expansion'] = df['1d_atr_14'] / df['1d_atr_14'].shift(5) if '1d_atr_14' in df.columns else 1
        if '1d_rsi_14' in df.columns:
            df['rsi_slope_5'] = df['1d_rsi_14'].diff(5)
            df['price_slope_5'] = df['close'].pct_change(5) * 100
            df['rsi_price_divergence'] = np.where((df['price_slope_5'] > 0) & (df['rsi_slope_5'] < 0), 1, np.where((df['price_slope_5'] < 0) & (df['rsi_slope_5'] > 0), -1, 0))
        return df

    def compute_live_features(self, crypto_id: str, feature_cols: list, scaler, seq_len: int = 30) -> Tuple[Optional[np.ndarray], Optional[pd.Series]]:
        try:
            df_1d = self._download_ohlcv(crypto_id, '1d', 300)
            df_1d = self._create_indicators(df_1d, '1d_')

            for tf in ['4h', '1w']:
                df_tf = self._download_ohlcv(crypto_id, tf, 300 if tf == '4h' else 100)
                df_tf = self._create_indicators(df_tf, f'{tf}_')
                tf_cols = ['date'] + [c for c in df_tf.columns if c.startswith(f'{tf}_')]
                df_1d = pd.merge_asof(df_1d.sort_values('date'), df_tf[tf_cols].sort_values('date'), on='date', direction='backward')

            df_1d = self._add_cross_tf_and_non_tech(df_1d)

            # Add regime + bear features if needed
            has_regime = any(c in feature_cols for c in ['sma_50', 'regime_bull', 'accumulation_score'])
            if has_regime:
                df_1d = self._add_market_regime_features(df_1d)
            has_bear = any(c in feature_cols for c in ['price_above_sma10_pct', 'roc_5', 'consec_red'])
            if has_bear:
                df_1d = self._add_bear_features(df_1d)

            for c in feature_cols:
                if c not in df_1d.columns: df_1d[c] = 0
                df_1d[c] = pd.to_numeric(df_1d[c], errors='coerce')
            df_1d[feature_cols] = df_1d[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

            raw_row = df_1d.iloc[-1]
            feat = df_1d[feature_cols].values.astype(np.float32)
            scaled = np.clip(np.nan_to_num(scaler.transform(feat), nan=0, posinf=0, neginf=0), -5, 5)
            if len(scaled) < seq_len:
                return None, None
            return scaled[-seq_len:], raw_row
        except Exception as e:
            logger.error(f"Feature error {crypto_id}: {e}")
            import traceback; traceback.print_exc()
            return None, None

    def _check_filters(self, raw_row, direction: str) -> Tuple[bool, str]:
        if direction == 'LONG':
            if 'distance_from_sma50' in raw_row.index and pd.notna(raw_row['distance_from_sma50']):
                if raw_row['distance_from_sma50'] < -0.05:
                    return False, "bear_market_sma50"
            if 'distance_from_sma20' in raw_row.index and pd.notna(raw_row['distance_from_sma20']):
                if raw_row['distance_from_sma20'] < -0.03:
                    return False, "bear_market_sma20"
            if 'trend_score' in raw_row.index and pd.notna(raw_row['trend_score']):
                if raw_row['trend_score'] < -3:
                    return False, "downtrend"
        else:
            if 'distance_from_sma50' in raw_row.index and pd.notna(raw_row['distance_from_sma50']):
                if raw_row['distance_from_sma50'] > 0.05:
                    return False, "bull_market"
            if 'trend_score' in raw_row.index and pd.notna(raw_row['trend_score']):
                if raw_row['trend_score'] > 3:
                    return False, "uptrend"
        return True, "pass"

    def _build_meta_features(self, crypto_id, raw_row, l_conf, l_dir, s_conf, s_dir, l_probs, s_probs):
        """Build meta-model input — matches training exactly"""
        if crypto_id not in self.meta_features:
            return None
        meta_feat_cols = self.meta_features[crypto_id]
        mf = {}
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
        for c in meta_feat_cols:
            if c not in mf:
                val = raw_row.get(c, 0) if hasattr(raw_row, 'get') else (raw_row[c] if c in raw_row.index else 0)
                mf[c] = float(val) if pd.notna(val) else 0.0
        return np.array([[mf.get(c, 0) for c in meta_feat_cols]])

    def _get_dynamic_tp_sl(self, raw_row, price: float, direction: str, crypto_id: str = '') -> Dict:
        atr = None
        if '1d_atr_14' in raw_row.index and pd.notna(raw_row['1d_atr_14']):
            atr = raw_row['1d_atr_14']

        # BTC: ATR symmetric (matches training)
        if crypto_id == 'bitcoin':
            ATR_MULT = 1.5
            if atr and atr > 0:
                tp_m = min(max(ATR_MULT * atr / price, 0.008), 0.04)
                sl_m = tp_m  # Symmetric
            else:
                tp_m, sl_m = 0.012, 0.012
            if direction == 'LONG':
                return {'target_price': round(price * (1 + tp_m), 2), 'stop_loss': round(price * (1 - sl_m), 2),
                        'take_profit_pct': round(tp_m * 100, 2), 'stop_loss_pct': round(sl_m * 100, 2), 'risk_reward_ratio': 1.0}
            else:
                return {'target_price': round(price * (1 - tp_m), 2), 'stop_loss': round(price * (1 + sl_m), 2),
                        'take_profit_pct': round(tp_m * 100, 2), 'stop_loss_pct': round(sl_m * 100, 2), 'risk_reward_ratio': 1.0}

        # Other coins: legacy asymmetric
        if direction == 'LONG':
            if atr and atr > 0:
                tp_m = min(max(atr / price, 0.008), 0.03)
                sl_m = tp_m * 0.5
            else:
                tp_m, sl_m = 0.015, 0.0075
            return {'target_price': round(price * (1 + tp_m), 2), 'stop_loss': round(price * (1 - sl_m), 2),
                    'take_profit_pct': round(tp_m * 100, 2), 'stop_loss_pct': round(sl_m * 100, 2), 'risk_reward_ratio': round(tp_m / sl_m, 2)}
        else:
            if atr and atr > 0:
                tp_m = min(max(atr / price, 0.01), 0.04)
                sl_m = tp_m * 0.5
            else:
                tp_m, sl_m = 0.02, 0.01
            return {'target_price': round(price * (1 - tp_m), 2), 'stop_loss': round(price * (1 + sl_m), 2),
                    'take_profit_pct': round(tp_m * 100, 2), 'stop_loss_pct': round(sl_m * 100, 2), 'risk_reward_ratio': round(tp_m / sl_m, 2)}

    async def predict_one(self, crypto_id: str) -> Dict:
        if crypto_id not in COIN_CONFIG:
            raise ValueError(f"Unknown crypto: {crypto_id}")

        cfg = COIN_CONFIG[crypto_id]
        price = self.get_live_price(crypto_id)

        # LONG prediction
        long_dir_val, long_conf, long_probs = None, None, None
        long_filter_reason = None
        raw_row = None
        if crypto_id in self.long_models and crypto_id in self.long_features:
            long_sl = self.long_seq_lens.get(crypto_id, 30)
            scaled, raw_row = self.compute_live_features(crypto_id, self.long_features[crypto_id], self.long_scalers[crypto_id], seq_len=long_sl)
            if scaled is not None:
                X = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)
                temp = self.long_temps.get(crypto_id, 1.0)
                with torch.no_grad():
                    logits = self.long_models[crypto_id](X)
                    probs = F.softmax(logits / temp, dim=1).squeeze()
                long_conf = probs.max(0)[0].item()
                long_dir_val = probs.max(0)[1].item()
                long_probs = probs.cpu().numpy()

        # SHORT prediction
        short_dir_val, short_conf, short_probs = None, None, None
        short_filter_reason = None
        if crypto_id in self.short_models and crypto_id in self.short_features:
            short_sl = self.short_seq_lens.get(crypto_id, 30)
            scaled_s, raw_row_s = self.compute_live_features(crypto_id, self.short_features[crypto_id], self.short_scalers[crypto_id], seq_len=short_sl)
            if scaled_s is not None:
                X_s = torch.tensor(scaled_s, dtype=torch.float32).unsqueeze(0)
                temp = self.short_temps.get(crypto_id, 1.0)
                with torch.no_grad():
                    logits = self.short_models[crypto_id](X_s)
                    probs = F.softmax(logits / temp, dim=1).squeeze()
                short_conf = probs.max(0)[0].item()
                short_dir_val = probs.max(0)[1].item()
                short_probs = probs.cpu().numpy()
                if raw_row is None:
                    raw_row = raw_row_s

        # Build meta features
        meta_input = None
        meta_long_prob, meta_short_prob = None, None
        if long_probs is not None and short_probs is not None and raw_row is not None:
            meta_input = self._build_meta_features(
                crypto_id, raw_row,
                long_conf or 0, long_dir_val or 0,
                short_conf or 0, short_dir_val or 0,
                long_probs, short_probs
            )

        # Determine signals with meta check
        long_signal = None
        if long_dir_val == 1 and long_conf is not None and long_conf >= cfg['long_conf']:
            if raw_row is not None:
                passes, reason = self._check_filters(raw_row, 'LONG')
                if passes:
                    # Meta check
                    meta_ok = True
                    if meta_input is not None and crypto_id in self.meta_long_models:
                        meta_long_prob = self.meta_long_models[crypto_id].predict_proba(meta_input)[0][1]
                        meta_ok = meta_long_prob >= cfg.get('long_meta_conf', 0.0)
                    if meta_ok:
                        long_signal = 'BUY'
                    else:
                        long_filter_reason = f"meta_blocked ({meta_long_prob:.1%})"
                else:
                    long_filter_reason = reason

        short_signal = None
        if short_dir_val == 1 and short_conf is not None and short_conf >= cfg['short_conf']:
            row = raw_row
            if row is not None:
                passes, reason = self._check_filters(row, 'SHORT')
                if passes:
                    meta_ok = True
                    if meta_input is not None and crypto_id in self.meta_short_models:
                        meta_short_prob = self.meta_short_models[crypto_id].predict_proba(meta_input)[0][1]
                        meta_ok = meta_short_prob >= cfg.get('short_meta_conf', 0.0)
                    if meta_ok:
                        short_signal = 'SELL'
                    else:
                        short_filter_reason = f"meta_blocked ({meta_short_prob:.1%})"
                else:
                    short_filter_reason = reason

        # Final signal (LONG priority)
        if long_signal:
            signal, confidence, direction = 'BUY', long_conf, 'LONG'
        elif short_signal:
            signal, confidence, direction = 'SELL', short_conf, 'SHORT'
        else:
            signal, direction = 'HOLD', None
            lc = long_conf if long_conf is not None else 0
            sc = short_conf if short_conf is not None else 0
            confidence = max(lc, sc)

        # Risk management
        risk_management = None
        if direction and price and raw_row is not None:
            risk_management = self._get_dynamic_tp_sl(raw_row, price, direction, crypto_id)

        symbol_map = {'bitcoin': 'BTCUSDT', 'ethereum': 'ETHUSDT', 'solana': 'SOLUSDT',
                      'dogecoin': 'DOGEUSDT', 'avalanche': 'AVAXUSDT', 'xrp': 'XRPUSDT',
                      'chainlink': 'LINKUSDT', 'cardano': 'ADAUSDT', 'near': 'NEARUSDT'}

        return {
            "crypto": crypto_id,
            "symbol": symbol_map.get(crypto_id, f'{crypto_id.upper()}USDT'),
            "name": crypto_id.capitalize(),
            "signal": signal,
            "direction": direction,
            "confidence": round(confidence, 4),
            "long_confidence": round(long_conf, 4) if long_conf is not None else None,
            "short_confidence": round(short_conf, 4) if short_conf is not None else None,
            "long_filter": long_filter_reason,
            "short_filter": short_filter_reason,
            "meta_long_prob": round(meta_long_prob, 4) if meta_long_prob is not None else None,
            "meta_short_prob": round(meta_short_prob, 4) if meta_short_prob is not None else None,
            "current_price": round(price, 4) if price else None,
            "risk_management": risk_management,
            "model": "CNN_1D_MultiScale + XGBoost_Meta",
            "features": "multi_tf_4h_1d_1w + market_regime",
            "timestamp": datetime.now().isoformat(),
            "data_source": "binance_live"
        }

    @property
    def models(self):
        return {**{k: v for k, v in self.long_models.items()}, **{k: v for k, v in self.short_models.items()}}
