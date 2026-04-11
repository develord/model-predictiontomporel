"""
Prediction Service
==================
Service pour générer les prédictions avec les modèles XGBoost V6
"""
import sys
import os
from pathlib import Path
import pickle
import numpy as np
from datetime import datetime
from typing import Dict, List
import logging

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import settings
from models import PredictionResponse, Probabilities, RiskManagement
# Removed train_models_xgboost import to avoid tensorflow dependency
# from train_models_xgboost import download_historical_data, calculate_indicators, prepare_features
from data_manager import get_historical_data
from xgboost_features import calculate_all_xgboost_features
from xgboost_features_v6 import calculate_all_xgboost_features_v6
from feature_selection_v5 import SELECTED_FEATURES_V5

import requests

# Alias for compatibility
download_historical_data = get_historical_data

# ============================================================================
# BASIC INDICATOR FUNCTIONS (No TensorFlow dependency)
# ============================================================================

def calculate_rsi(closes, period=14):
    """Calculate RSI indicator"""
    if len(closes) < period + 1:
        return None

    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_ema(closes, period):
    """Calculate EMA indicator"""
    if len(closes) < period:
        return None

    multiplier = 2 / (period + 1)
    ema = np.mean(closes[:period])

    for price in closes[period:]:
        ema = (price * multiplier) + (ema * (1 - multiplier))

    return ema

def calculate_macd(closes, fast=12, slow=26, signal=9):
    """Calculate MACD indicator"""
    if len(closes) < slow + signal:
        return None

    ema_fast = calculate_ema(closes, fast)
    ema_slow = calculate_ema(closes, slow)

    if ema_fast is None or ema_slow is None:
        return None

    macd_line = ema_fast - ema_slow

    # Calculate signal line (EMA of MACD)
    macd_values = []
    for i in range(slow, len(closes)):
        window = closes[:i+1]
        fast_ema = calculate_ema(window, fast)
        slow_ema = calculate_ema(window, slow)
        if fast_ema is not None and slow_ema is not None:
            macd_values.append(fast_ema - slow_ema)

    if len(macd_values) < signal:
        signal_line = macd_line
    else:
        signal_line = calculate_ema(macd_values, signal)

    histogram = macd_line - (signal_line if signal_line else 0)

    return {
        'macd': macd_line,
        'signal': signal_line if signal_line else 0,
        'histogram': histogram
    }

def calculate_bollinger_bands(closes, period=20, std_dev=2):
    """Calculate Bollinger Bands"""
    if len(closes) < period:
        return None

    sma = np.mean(closes[-period:])
    std = np.std(closes[-period:])

    return {
        'upper': sma + (std * std_dev),
        'middle': sma,
        'lower': sma - (std * std_dev)
    }

def calculate_stochastic_rsi(closes, period=14, k_period=3, d_period=3):
    """Calculate Stochastic RSI"""
    if len(closes) < period * 2:
        return None

    # Calculate RSI values for the period
    rsi_values = []
    for i in range(period, len(closes) + 1):
        window = closes[:i]
        rsi = calculate_rsi(window, period)
        if rsi is not None:
            rsi_values.append(rsi)

    if len(rsi_values) < period:
        return {'k': 50, 'd': 50}

    # Calculate Stochastic of RSI
    recent_rsi = rsi_values[-period:]
    max_rsi = max(recent_rsi)
    min_rsi = min(recent_rsi)

    if max_rsi == min_rsi:
        stoch_rsi = 50
    else:
        stoch_rsi = ((rsi_values[-1] - min_rsi) / (max_rsi - min_rsi)) * 100

    # K and D are simplified (should use SMA of stoch_rsi)
    k = stoch_rsi
    d = stoch_rsi  # Simplified

    return {'k': k, 'd': d}

def calculate_atr(klines, period=14):
    """Calculate ATR indicator"""
    if len(klines) < period + 1:
        return 0

    true_ranges = []
    for i in range(1, len(klines)):
        high = float(klines[i][2])
        low = float(klines[i][3])
        prev_close = float(klines[i-1][4])

        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        )
        true_ranges.append(tr)

    if len(true_ranges) < period:
        return np.mean(true_ranges) if true_ranges else 0

    return np.mean(true_ranges[-period:])

def calculate_obv(klines):
    """Calculate OBV indicator"""
    if len(klines) < 2:
        return 0

    obv = 0
    for i in range(1, len(klines)):
        close = float(klines[i][4])
        prev_close = float(klines[i-1][4])
        volume = float(klines[i][5])

        if close > prev_close:
            obv += volume
        elif close < prev_close:
            obv -= volume

    return obv

def calculate_indicators(klines):
    """Calculate technical indicators without tensorflow dependency"""
    closes = [float(k[4]) for k in klines]
    highs = [float(k[2]) for k in klines]
    lows = [float(k[3]) for k in klines]
    volumes = [float(k[5]) for k in klines]
    current_price = closes[-1]

    return {
        'rsi': calculate_rsi(closes),
        'ema20': calculate_ema(closes, 20),
        'ema50': calculate_ema(closes, 50),
        'ema200': calculate_ema(closes, 200),
        'macd': calculate_macd(closes),
        'bollingerBands': calculate_bollinger_bands(closes),
        'stochasticRsi': calculate_stochastic_rsi(closes),
        'atr': calculate_atr(klines),
        'obv': calculate_obv(klines),
        'currentPrice': current_price,
        'high': highs[-1],
        'low': lows[-1],
        'volume': volumes[-1]
    }

def prepare_features(indicators, current_price, prices_history=None):
    """Prepare features for model prediction without tensorflow dependency"""
    features = []

    # RSI (2 features)
    if indicators['rsi'] is not None:
        features.append(indicators['rsi'] / 100)
        features.append((indicators['rsi'] - 50) / 50 if indicators['rsi'] > 50 else (50 - indicators['rsi']) / 50)
    else:
        features.extend([0.5, 0])

    # EMA features (3 features)
    for ema_key in ['ema20', 'ema50', 'ema200']:
        if indicators[ema_key] is not None:
            features.append((current_price - indicators[ema_key]) / current_price)
        else:
            features.append(0)

    # MACD (3 features)
    if indicators['macd'] is not None:
        macd_line = indicators['macd'].get('macd', 0)
        signal_line = indicators['macd'].get('signal', 0)
        histogram = indicators['macd'].get('histogram', 0)
        features.extend([macd_line / current_price if current_price > 0 else 0,
                        signal_line / current_price if current_price > 0 else 0,
                        histogram / current_price if current_price > 0 else 0])
    else:
        features.extend([0, 0, 0])

    # Bollinger Bands (2 features)
    if indicators['bollingerBands'] is not None:
        upper = indicators['bollingerBands'].get('upper', current_price)
        lower = indicators['bollingerBands'].get('lower', current_price)
        features.extend([(current_price - lower) / (upper - lower) if upper != lower else 0.5,
                        (upper - lower) / current_price if current_price > 0 else 0])
    else:
        features.extend([0.5, 0])

    # Stochastic RSI (2 features)
    if indicators['stochasticRsi'] is not None:
        features.extend([indicators['stochasticRsi'].get('k', 50) / 100,
                        indicators['stochasticRsi'].get('d', 50) / 100])
    else:
        features.extend([0.5, 0.5])

    # ATR & OBV (2 features)
    features.append(indicators.get('atr', 0) / current_price if current_price > 0 else 0)
    features.append(np.tanh(indicators.get('obv', 0) / 1e9))  # Normalize OBV

    # Price momentum features (11 features - Phase 1)
    if prices_history and len(prices_history) >= 20:
        for period in [1, 3, 7, 14, 30]:
            if len(prices_history) > period:
                features.append((prices_history[-1] - prices_history[-period-1]) / prices_history[-period-1])
            else:
                features.append(0)

        # Volatility
        returns = [(prices_history[i] - prices_history[i-1]) / prices_history[i-1]
                  for i in range(max(1, len(prices_history)-20), len(prices_history))]
        features.append(np.std(returns) if returns else 0)

        # Price position in 20-day range
        recent_prices = prices_history[-20:]
        price_range = max(recent_prices) - min(recent_prices)
        features.append((current_price - min(recent_prices)) / price_range if price_range > 0 else 0.5)

        # Moving averages crossover signals
        if len(prices_history) >= 50:
            ma20 = np.mean(prices_history[-20:])
            ma50 = np.mean(prices_history[-50:])
            features.extend([(ma20 - ma50) / ma50 if ma50 > 0 else 0,
                           1 if ma20 > ma50 else 0,
                           (current_price - ma20) / ma20 if ma20 > 0 else 0])
        else:
            features.extend([0, 0, 0])
    else:
        features.extend([0] * 11)

    return features[:29]  # Return 29 base features

logger = logging.getLogger(__name__)


class PredictionService:
    """Service for ML predictions"""

    CRYPTO_INFO = {
        'bitcoin': {'symbol': 'BTCUSDT', 'name': 'Bitcoin'},
        'ethereum': {'symbol': 'ETHUSDT', 'name': 'Ethereum'},
        'bnb': {'symbol': 'BNBUSDT', 'name': 'BNB'},
        'xrp': {'symbol': 'XRPUSDT', 'name': 'XRP'},
        'cardano': {'symbol': 'ADAUSDT', 'name': 'Cardano'},
        'avalanche': {'symbol': 'AVAXUSDT', 'name': 'Avalanche'},
        'polkadot': {'symbol': 'DOTUSDT', 'name': 'Polkadot'},
        'solana': {'symbol': 'SOLUSDT', 'name': 'Solana'}
    }

    def __init__(self):
        self.models = {}
        self.indicators_history = {}

    async def load_models(self):
        """Load all XGBoost V6 models"""
        logger.info("Loading XGBoost V6 models...")

        for crypto_id in self.CRYPTO_INFO.keys():
            # Solana uses optimized model (V6 standard gave mediocre results)
            if crypto_id == 'solana':
                model_path = settings.MODELS_DIR / f"{crypto_id}_1d_xgboost_v6_optimized.pkl"
            else:
                model_path = settings.MODELS_DIR / f"{crypto_id}_1d_xgboost_v6.pkl"

            if not model_path.exists():
                logger.warning(f"Model not found: {model_path}")
                continue

            try:
                with open(model_path, 'rb') as f:
                    self.models[crypto_id] = pickle.load(f)
                logger.info(f"✅ Loaded {crypto_id} model")
            except Exception as e:
                logger.error(f"Failed to load {crypto_id} model: {e}")

        logger.info(f"Loaded {len(self.models)}/{len(self.CRYPTO_INFO)} models")

        if len(self.models) == 0:
            raise Exception("No models loaded!")

    def select_features_from_vector(self, features_69, selected_features_names):
        """Extract selected features from full 69-feature vector"""
        return features_69[:41]

    def prepare_features_v6(self, crypto_id: str, klines, klines_btc=None):
        """Prepare V6 features (41 V5 + 16 V6 = 57 features)"""
        if len(klines) < 201:
            raise ValueError("Need at least 201 candles")

        # Initialize indicators history if needed
        if crypto_id not in self.indicators_history:
            self.indicators_history[crypto_id] = []

        # Use last 201 candles
        i = len(klines) - 1
        window_data = klines[max(0, i-200):i+1]
        indicators = calculate_indicators(window_data)
        current_price = float(klines[i][4])

        # Update indicators history
        self.indicators_history[crypto_id].append(indicators)
        if len(self.indicators_history[crypto_id]) > 10:
            self.indicators_history[crypto_id].pop(0)

        prices_history = [float(k[4]) for k in window_data]
        features_base = prepare_features(indicators, current_price, prices_history)

        # BTC data
        window_data_btc = None
        if klines_btc and len(klines_btc) >= i+1:
            window_data_btc = klines_btc[max(0, i-200):i+1]

        volumes = [float(k[5]) for k in window_data]

        # V5 features (69)
        features_xgb_v5 = calculate_all_xgboost_features(
            window_data,
            indicators,
            volumes,
            crypto_symbol=self.CRYPTO_INFO[crypto_id]['symbol'],
            klines_btc=window_data_btc,
            indicators_history=self.indicators_history[crypto_id]
        )

        # V6 features (16)
        features_xgb_v6 = calculate_all_xgboost_features_v6(
            window_data,
            indicators,
            volumes,
            crypto_symbol=self.CRYPTO_INFO[crypto_id]['symbol'],
            klines_btc=window_data_btc,
            indicators_history=self.indicators_history[crypto_id]
        )

        # Combine: base + V5 + V6
        features_69_v5 = features_base + features_xgb_v5
        features_selected_v5 = self.select_features_from_vector(features_69_v5, SELECTED_FEATURES_V5)
        features_all_v6 = features_selected_v5 + features_xgb_v6

        return np.array(features_all_v6), current_price

    def calculate_risk_management(self, signal: str, current_price: float, confidence: float) -> RiskManagement:
        """
        Calculate risk management metrics based on signal and confidence

        Logic:
        - BUY: Target above current, Stop Loss below
        - SELL: Target below current, Stop Loss above
        - HOLD: No risk management (None)
        """
        if signal == "HOLD":
            return None

        # Parameters
        STOP_LOSS_PERCENT = 0.02  # 2% stop loss
        MAX_TARGET_PERCENT = 0.15  # Maximum 15% target

        if signal == "BUY":
            # Calculate target based on confidence (higher confidence = higher target)
            target_percent = confidence * MAX_TARGET_PERCENT
            target_price = current_price * (1 + target_percent)
            stop_loss = current_price * (1 - STOP_LOSS_PERCENT)
            take_profit = target_price

            # Calculate percentages
            potential_gain_percent = ((take_profit - current_price) / current_price) * 100
            potential_loss_percent = ((current_price - stop_loss) / current_price) * 100

            # Risk:Reward = Gain / Loss
            risk_reward_ratio = (take_profit - current_price) / (current_price - stop_loss)

        else:  # SELL
            # Calculate target based on confidence (higher confidence = lower target)
            target_percent = confidence * MAX_TARGET_PERCENT
            target_price = current_price * (1 - target_percent)
            stop_loss = current_price * (1 + STOP_LOSS_PERCENT)
            take_profit = target_price

            # Calculate percentages
            potential_gain_percent = ((current_price - take_profit) / current_price) * 100
            potential_loss_percent = ((stop_loss - current_price) / current_price) * 100

            # Risk:Reward = Gain / Loss
            risk_reward_ratio = (current_price - take_profit) / (stop_loss - current_price)

        return RiskManagement(
            target_price=round(target_price, 2),
            stop_loss=round(stop_loss, 2),
            take_profit=round(take_profit, 2),
            risk_reward_ratio=round(risk_reward_ratio, 2),
            potential_gain_percent=round(potential_gain_percent, 2),
            potential_loss_percent=round(potential_loss_percent, 2)
        )

    async def get_current_price(self, symbol: str) -> float:
        """Get current price from Binance"""
        # Use cached data for now
        klines = download_historical_data(symbol, interval='1d', limit=1)
        if klines and len(klines) > 0:
            return float(klines[-1][4])
        raise Exception(f"Failed to get price for {symbol}")

    async def predict_one(self, crypto_id: str) -> PredictionResponse:
        """Generate prediction for one crypto"""
        if crypto_id not in self.models:
            raise ValueError(f"Model not loaded for {crypto_id}")

        model = self.models[crypto_id]
        crypto_info = self.CRYPTO_INFO[crypto_id]
        symbol = crypto_info['symbol']

        # Download data
        klines = download_historical_data(symbol, interval='1d', limit=300)

        klines_btc = None
        if symbol != 'BTCUSDT':
            klines_btc = download_historical_data('BTCUSDT', interval='1d', limit=300)

        # Prepare features
        features, current_price = self.prepare_features_v6(crypto_id, klines, klines_btc)

        # Predict
        features_reshaped = features.reshape(1, -1)
        prediction = model.predict(features_reshaped)[0]
        probabilities = model.predict_proba(features_reshaped)[0]

        # Map prediction to signal
        signal_map = {0: "BUY", 1: "SELL", 2: "HOLD"}
        signal = signal_map[prediction]
        confidence = float(probabilities.max())

        # Calculate risk management
        risk_mgmt = self.calculate_risk_management(signal, current_price, confidence)
        logger.info(f"Risk management for {crypto_id}: {risk_mgmt}")

        return PredictionResponse(
            crypto=crypto_id,
            symbol=symbol,
            name=crypto_info['name'],
            signal=signal,
            confidence=confidence,
            probabilities=Probabilities(
                buy=float(probabilities[0]),
                sell=float(probabilities[1]),
                hold=float(probabilities[2])
            ),
            current_price=current_price,
            risk_management=risk_mgmt,
            timestamp=datetime.now().isoformat()
        )

    async def predict_all(self) -> Dict[str, PredictionResponse]:
        """Generate predictions for all cryptos"""
        predictions = {}

        for crypto_id in self.models.keys():
            try:
                prediction = await self.predict_one(crypto_id)
                predictions[crypto_id] = prediction
            except Exception as e:
                logger.error(f"Error predicting {crypto_id}: {e}")
                continue

        return predictions

