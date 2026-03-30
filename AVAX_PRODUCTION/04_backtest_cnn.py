"""
AVAX CNN Backtesting - Realistic Intraday with Intelligent Filters
=====================================================================
- 15min candle monitoring for TP/SL
- Intelligent signal filtering (volatility, momentum alignment, consecutive loss break)
- CNN model from {crypto}_cnn_model.pt

Usage:
    python 04_backtest_cnn.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import joblib
import json
import ccxt
import logging
from datetime import timedelta

CRYPTO = 'AVAX'
CRYPTO_LOWER = CRYPTO.lower()
PAIR = f'{CRYPTO}/USDT'

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR / 'scripts'))

from direction_prediction_model import CNNDirectionModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = BASE_DIR / 'data' / 'cache'
MODEL_DIR = BASE_DIR / 'models'
RESULTS_DIR = BASE_DIR / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Trading parameters
TP_PCT = 0.015
SL_PCT = 0.0075
POSITION_SIZE = 0.95
TRADING_FEE = 0.001
SLIPPAGE = 0.0005
CONFIDENCE_THRESHOLD = 0.55

# Intelligent filters
MAX_CONSECUTIVE_LOSSES = 2
COOLDOWN_DAYS = 5
MIN_MOMENTUM_ALIGNMENT = 1
MAX_VOLATILITY_REGIME = 2.5
USE_DYNAMIC_TP_SL = True

BACKTEST_START = '2026-01-01'
BACKTEST_END = '2026-03-24'


def load_model():
    """Load CNN model from checkpoint"""
    model_file = MODEL_DIR / f'{CRYPTO_LOWER}_cnn_model.pt'
    if not model_file.exists():
        logger.error(f"Model not found: {model_file}")
        return None, None

    checkpoint = torch.load(model_file, map_location='cpu', weights_only=False)

    model_type = checkpoint.get('model_type', 'cnn') if isinstance(checkpoint, dict) else 'cnn'
    feature_dim = checkpoint.get('feature_dim', 90) if isinstance(checkpoint, dict) else 90
    seq_len = checkpoint.get('sequence_length', 60) if isinstance(checkpoint, dict) else 60

    model = CNNDirectionModel(feature_dim=feature_dim, sequence_length=seq_len, dropout=0.35)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    logger.info(f"Model: {model_type} (seq={seq_len}, feat={feature_dim})")
    return model, seq_len


def fetch_15min_data(start_date, end_date):
    """Fetch 15min candles from Binance"""
    cache_file = DATA_DIR / f'{CRYPTO_LOWER}_15min_q1_2026.csv'

    if cache_file.exists():
        df = pd.read_csv(cache_file, parse_dates=['timestamp'])
        df.set_index('timestamp', inplace=True)
        logger.info(f"15min cache: {len(df)} candles")
        return df

    logger.info("Downloading 15min candles from Binance...")
    exchange = ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'spot'}})

    start_ts = int(pd.Timestamp(start_date).timestamp() * 1000)
    end_ts = int(pd.Timestamp(end_date).timestamp() * 1000)
    all_candles = []
    current_ts = start_ts

    while current_ts < end_ts:
        candles = exchange.fetch_ohlcv(PAIR, '15m', since=current_ts, limit=1000)
        if not candles:
            break
        all_candles.extend(candles)
        current_ts = candles[-1][0] + 1
        if len(all_candles) % 3000 == 0:
            logger.info(f"  {len(all_candles)} candles...")

    df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[(df.index >= start_date) & (df.index < end_date)]
    df.to_csv(cache_file)
    logger.info(f"15min data: {len(df)} candles")
    return df


def get_dynamic_tp_sl(row, entry_price):
    """Calculate dynamic TP/SL based on ATR if available"""
    if not USE_DYNAMIC_TP_SL:
        return entry_price * (1 + TP_PCT), entry_price * (1 - SL_PCT)

    atr = row.get('1d_atr_14', None)
    if atr is not None and pd.notna(atr) and atr > 0:
        tp_mult = min(max(atr / entry_price, 0.008), 0.03)
        sl_mult = tp_mult * 0.5
        return entry_price * (1 + tp_mult), entry_price * (1 - sl_mult)

    return entry_price * (1 + TP_PCT), entry_price * (1 - SL_PCT)


def simulate_trade_intraday(entry_price, entry_date, df_15m, tp_price, sl_price, max_hold_days=10):
    """Simulate trade with 15min candles and dynamic TP/SL"""

    for day_offset in range(1, max_hold_days + 1):
        check_day = entry_date + timedelta(days=day_offset)
        check_day_end = check_day + timedelta(days=1)
        intraday = df_15m[(df_15m.index >= check_day) & (df_15m.index < check_day_end)]

        for ts, candle in intraday.iterrows():
            if candle['high'] >= tp_price:
                return {'exit_type': 'TP', 'exit_price': tp_price * (1 - SLIPPAGE),
                        'exit_time': ts, 'duration_min': (ts - entry_date).total_seconds() / 60}
            if candle['low'] <= sl_price:
                return {'exit_type': 'SL', 'exit_price': sl_price * (1 - SLIPPAGE),
                        'exit_time': ts, 'duration_min': (ts - entry_date).total_seconds() / 60}

    last_check = entry_date + timedelta(days=max_hold_days)
    remaining = df_15m[df_15m.index <= last_check]
    final_price = remaining.iloc[-1]['close'] if len(remaining) > 0 else entry_price
    return {'exit_type': 'EOD', 'exit_price': final_price,
            'exit_time': last_check, 'duration_min': max_hold_days * 1440}


def check_signal_filters(row, df_features, idx, consecutive_losses, cooldown_until, current_date, confidence=0.5):
    """Intelligent signal filtering"""

    # 1. Cooldown after consecutive losses
    if cooldown_until is not None and current_date <= cooldown_until:
        return False, f"cooldown_until_{cooldown_until.date()}"

    # 2. Momentum alignment
    bullish_count = 0
    total_tf = 0
    for col in ['1d_momentum_5', '4h_momentum_5', '1w_momentum_5']:
        if col in row.index and pd.notna(row[col]):
            total_tf += 1
            if row[col] > 0:
                bullish_count += 1
    if total_tf > 0 and bullish_count < MIN_MOMENTUM_ALIGNMENT:
        return False, f"weak_momentum_{bullish_count}/{total_tf}"

    # 3. Volatility regime filter
    if 'volatility_regime' in row.index and pd.notna(row['volatility_regime']):
        if row['volatility_regime'] > MAX_VOLATILITY_REGIME:
            return False, f"high_volatility_{row['volatility_regime']:.2f}"

    # 4. ADX filter - avoid ranging markets
    if '1d_adx_14' in row.index and pd.notna(row['1d_adx_14']):
        if row['1d_adx_14'] < 15:
            return False, "no_trend_adx"

    # 5. Bear market filter - don't go LONG if price well below SMA50
    if 'distance_from_sma50' in row.index and pd.notna(row['distance_from_sma50']):
        if row['distance_from_sma50'] < -0.05:
            return False, "bear_market_sma50"

    # 5b. Short-term trend - don't go LONG if price below SMA20
    if 'distance_from_sma20' in row.index and pd.notna(row['distance_from_sma20']):
        if row['distance_from_sma20'] < -0.03:
            return False, "bear_market_sma20"

    # 6. Require higher confidence when volatility is high
    if 'volatility_regime' in row.index and pd.notna(row['volatility_regime']):
        if row['volatility_regime'] > 1.5 and confidence < 0.65:
            return False, "low_conf_high_vol"

    # 7. RSI divergence - don't buy if RSI overbought on multiple TFs
    overbought = 0
    for col in ['1d_rsi_14', '4h_rsi_14', '1w_rsi_14']:
        if col in row.index and pd.notna(row[col]) and row[col] > 70:
            overbought += 1
    if overbought >= 2:
        return False, "overbought_multi_tf"

    # 8. Trend score filter - avoid downtrends
    if 'trend_score' in row.index and pd.notna(row['trend_score']):
        if row['trend_score'] < -3:
            return False, "downtrend"

    return True, "pass"


def backtest():
    """Full backtest with intelligent filters + 15min monitoring"""
    logger.info(f"\n{'='*70}")
    logger.info(f"{CRYPTO} CNN BACKTEST - REALISTIC + INTELLIGENT FILTERS (Q1 2026)")
    logger.info(f"{'='*70}\n")

    model, seq_len = load_model()
    if model is None:
        return

    # Load data
    df = pd.read_csv(DATA_DIR / f'{CRYPTO_LOWER}_features.csv')
    df['date'] = pd.to_datetime(df['date'])

    with open(MODEL_DIR / 'avax_cnn_features.json', 'r') as f:
        feature_cols = json.load(f)

    scaler = None
    scaler_path = MODEL_DIR / 'feature_scaler.joblib'
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)

    df_wide = df[df['date'] >= '2025-01-01'].copy()
    all_features = df_wide[feature_cols].fillna(0).values
    if scaler:
        all_features = np.nan_to_num(scaler.transform(all_features), nan=0.0, posinf=0.0, neginf=0.0)

    df_test = df_wide[(df_wide['date'] >= BACKTEST_START) & (df_wide['date'] <= BACKTEST_END)].copy()
    test_start_idx = len(df_wide) - len(df_test)

    logger.info(f"Test: {len(df_test)} days | Features: {len(feature_cols)} | Seq: {seq_len}")

    # 15min candles
    df_15m = fetch_15min_data(BACKTEST_START, BACKTEST_END)

    # Trading
    capital = 1000.0
    trades = []
    consecutive_losses = 0
    cooldown_until = None

    total_signals = 0
    filtered_signals = 0
    filter_reasons = {}

    logger.info(f"Confidence: >{CONFIDENCE_THRESHOLD} | Max consec losses: {MAX_CONSECUTIVE_LOSSES}")
    logger.info(f"Cooldown: {COOLDOWN_DAYS}d | Min momentum align: {MIN_MOMENTUM_ALIGNMENT}\n")

    for i in range(len(df_test)):
        wide_idx = test_start_idx + i
        row = df_test.iloc[i]
        current_date = row['date']

        if wide_idx < seq_len:
            continue

        # Predict
        seq = all_features[wide_idx-seq_len:wide_idx]
        X = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            direction, confidence = model.predict_direction(X)

        d = direction.item()
        c = confidence.item()

        if d != 1 or c <= CONFIDENCE_THRESHOLD:
            continue

        total_signals += 1

        # Apply filters
        passes, reason = check_signal_filters(
            df_test.iloc[i], df, wide_idx, consecutive_losses, cooldown_until, current_date, c
        )

        if not passes:
            filtered_signals += 1
            filter_reasons[reason] = filter_reasons.get(reason, 0) + 1
            continue

        # Execute trade with dynamic TP/SL
        entry_price = row['close'] * (1 + SLIPPAGE)
        tp_price, sl_price = get_dynamic_tp_sl(df_test.iloc[i], entry_price)
        position_value = capital * POSITION_SIZE
        shares = position_value / entry_price
        capital -= position_value + (position_value * TRADING_FEE)

        result = simulate_trade_intraday(entry_price, current_date, df_15m, tp_price, sl_price)

        exit_value = shares * result['exit_price']
        capital += exit_value - (exit_value * TRADING_FEE)
        pnl_pct = (result['exit_price'] / entry_price - 1) * 100

        trades.append({
            'entry_date': current_date,
            'entry_price': entry_price,
            'confidence': c,
            'exit_type': result['exit_type'],
            'exit_price': result['exit_price'],
            'exit_time': result['exit_time'],
            'pnl_pct': pnl_pct,
            'duration_h': result['duration_min'] / 60
        })

        # Update consecutive loss tracking
        if result['exit_type'] == 'SL':
            consecutive_losses += 1
            if consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
                cooldown_until = current_date + timedelta(days=COOLDOWN_DAYS)
                logger.info(f"  >>> {MAX_CONSECUTIVE_LOSSES} consecutive losses -> cooldown until {cooldown_until.date()}")
        else:
            consecutive_losses = 0

        dur_h = result['duration_min'] / 60
        logger.info(f"  {current_date.date()} | Conf:{c:.1%} | {result['exit_type']} @ ${result['exit_price']:.4f} | "
                    f"PnL:{pnl_pct:+.2f}% | {dur_h:.0f}h | Capital:${capital:.0f}")

    # Results
    logger.info(f"\n{'='*70}")
    logger.info(f"RESULTS")
    logger.info(f"{'='*70}\n")

    logger.info(f"Signal Filtering:")
    logger.info(f"  Raw signals: {total_signals}")
    logger.info(f"  Filtered: {filtered_signals} ({filtered_signals/max(total_signals,1)*100:.0f}%)")
    logger.info(f"  Executed: {len(trades)}")
    if filter_reasons:
        for reason, count in sorted(filter_reasons.items(), key=lambda x: -x[1]):
            logger.info(f"    {reason}: {count}")

    if trades:
        trades_df = pd.DataFrame(trades)
        total_return = (capital / 1000 - 1) * 100
        tp_count = len(trades_df[trades_df['exit_type'] == 'TP'])
        sl_count = len(trades_df[trades_df['exit_type'] == 'SL'])
        eod_count = len(trades_df[trades_df['exit_type'] == 'EOD'])
        win_rate = tp_count / len(trades) * 100

        logger.info(f"\nTrading:")
        logger.info(f"  Total Trades: {len(trades)}")
        logger.info(f"  TP: {tp_count} | SL: {sl_count} | EOD: {eod_count}")
        logger.info(f"  Win Rate: {win_rate:.1f}%")
        logger.info(f"  Total Return: {total_return:+.2f}%")
        logger.info(f"  Final Capital: ${capital:.2f}")

        if tp_count > 0:
            logger.info(f"  Avg TP duration: {trades_df[trades_df['exit_type']=='TP']['duration_h'].mean():.0f}h")
        if sl_count > 0:
            logger.info(f"  Avg SL duration: {trades_df[trades_df['exit_type']=='SL']['duration_h'].mean():.0f}h")

        trades_df.to_csv(RESULTS_DIR / f'{CRYPTO_LOWER}_cnn_backtest_trades.csv', index=False)
        summary = {
            'type': 'cnn_realistic_15min_filtered',
            'crypto': CRYPTO,
            'period': f'{BACKTEST_START} to {BACKTEST_END}',
            'trades': len(trades), 'tp': tp_count, 'sl': sl_count, 'eod': eod_count,
            'win_rate': round(win_rate, 2),
            'return_pct': round(total_return, 2),
            'final_capital': round(capital, 2),
            'confidence_threshold': CONFIDENCE_THRESHOLD,
            'filters': filter_reasons
        }
        with open(RESULTS_DIR / f'{CRYPTO_LOWER}_cnn_backtest_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"\nResults saved to {RESULTS_DIR}")
    else:
        logger.info("\nNo trades executed")


if __name__ == "__main__":
    backtest()
