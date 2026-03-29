"""
SOL CNN Backtesting - Realistic 15min + Intelligent Filters
============================================================
Same approach as BTC CNN backtest that achieved 75% WR.

Usage:
    python 04_backtest_cnn.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import joblib
import json
import ccxt
import logging
from datetime import timedelta

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR / 'scripts'))

from direction_prediction_model import CNNDirectionModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = BASE_DIR / 'data' / 'cache'
MODEL_DIR = BASE_DIR / 'models'
RESULTS_DIR = BASE_DIR / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CRYPTO = 'SOL'
TP_PCT = 0.015
SL_PCT = 0.0075
POSITION_SIZE = 0.95
TRADING_FEE = 0.001
SLIPPAGE = 0.0005
CONFIDENCE_THRESHOLD = 0.65
USE_DYNAMIC_TP_SL = True

MAX_CONSECUTIVE_LOSSES = 2
COOLDOWN_DAYS = 5
MIN_MOMENTUM_ALIGNMENT = 1

BACKTEST_START = '2026-01-01'
BACKTEST_END = '2026-03-24'


def load_model():
    model_file = MODEL_DIR / f'{CRYPTO.lower()}_cnn_model.pt'
    if not model_file.exists():
        logger.error(f"Model not found: {model_file}")
        return None, None, None

    ckpt = torch.load(model_file, map_location='cpu', weights_only=False)
    feat_dim = ckpt.get('feature_dim', 384)
    seq_len = ckpt.get('sequence_length', 30)

    model = CNNDirectionModel(feature_dim=feat_dim, sequence_length=seq_len, dropout=0.4)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    logger.info(f"Model: CNN (feat={feat_dim}, seq={seq_len})")
    return model, seq_len, feat_dim


def fetch_15min_data():
    cache_file = DATA_DIR / f'{CRYPTO.lower()}_15min_q1_2026.csv'
    if cache_file.exists():
        df = pd.read_csv(cache_file, parse_dates=['timestamp'])
        df.set_index('timestamp', inplace=True)
        logger.info(f"15min cache: {len(df)} candles")
        return df

    logger.info(f"Downloading {CRYPTO} 15min candles...")
    exchange = ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'spot'}})
    start_ts = int(pd.Timestamp(BACKTEST_START).timestamp() * 1000)
    end_ts = int(pd.Timestamp(BACKTEST_END).timestamp() * 1000)
    all_candles = []
    current_ts = start_ts
    while current_ts < end_ts:
        candles = exchange.fetch_ohlcv(f'{CRYPTO}/USDT', '15m', since=current_ts, limit=1000)
        if not candles:
            break
        all_candles.extend(candles)
        current_ts = candles[-1][0] + 1

    df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[(df.index >= BACKTEST_START) & (df.index < BACKTEST_END)]
    df.to_csv(cache_file)
    logger.info(f"15min: {len(df)} candles")
    return df


def get_dynamic_tp_sl(row, entry_price):
    if not USE_DYNAMIC_TP_SL:
        return entry_price * (1 + TP_PCT), entry_price * (1 - SL_PCT)
    atr = None
    for col in ['1d_atr_14', '1d_atr_pct']:
        if col in row.index and pd.notna(row[col]):
            if col == '1d_atr_14':
                atr = row[col]
            break
    if atr is not None and atr > 0:
        tp_mult = min(max(atr / entry_price, 0.008), 0.03)
        sl_mult = tp_mult * 0.5
        return entry_price * (1 + tp_mult), entry_price * (1 - sl_mult)
    return entry_price * (1 + TP_PCT), entry_price * (1 - SL_PCT)


def simulate_trade_15min(entry_price, entry_date, df_15m, tp_price, sl_price, max_days=10):
    for day_off in range(1, max_days + 1):
        day = entry_date + timedelta(days=day_off)
        day_end = day + timedelta(days=1)
        intraday = df_15m[(df_15m.index >= day) & (df_15m.index < day_end)]
        for ts, c in intraday.iterrows():
            if c['high'] >= tp_price:
                return {'exit_type': 'TP', 'exit_price': tp_price * (1 - SLIPPAGE),
                        'exit_time': ts, 'duration_min': (ts - entry_date).total_seconds() / 60}
            if c['low'] <= sl_price:
                return {'exit_type': 'SL', 'exit_price': sl_price * (1 - SLIPPAGE),
                        'exit_time': ts, 'duration_min': (ts - entry_date).total_seconds() / 60}
    remaining = df_15m[df_15m.index <= entry_date + timedelta(days=max_days)]
    fp = remaining.iloc[-1]['close'] if len(remaining) > 0 else entry_price
    return {'exit_type': 'EOD', 'exit_price': fp, 'exit_time': entry_date + timedelta(days=max_days),
            'duration_min': max_days * 1440}


def check_filters(row, consecutive_losses, cooldown_until, current_date):
    if cooldown_until and current_date <= cooldown_until:
        return False, "cooldown"

    # Bear market SMA50 only (relaxed - CNN handles bear better)
    if '1d_sma_50' in row.index and pd.notna(row['1d_sma_50']) and row['1d_sma_50'] > 0:
        dist = (row['close'] / row['1d_sma_50']) - 1
        if dist < -0.10:  # Block if >10% below SMA50
            return False, "deep_bear"

    # SMA20 bear filter
    if '1d_sma_20' in row.index and pd.notna(row['1d_sma_20']) and row['1d_sma_20'] > 0:
        dist20 = (row['close'] / row['1d_sma_20']) - 1
        if dist20 < -0.05:
            return False, "bear_sma20"

    # BTC strongly bearish (trend alignment very negative)
    if '1d_btc_trend_alignment' in row.index and pd.notna(row['1d_btc_trend_alignment']):
        if row['1d_btc_trend_alignment'] < -1.0:
            return False, "btc_very_bearish"

    return True, "pass"


def backtest():
    logger.info(f"\n{'='*70}")
    logger.info(f"{CRYPTO} CNN BACKTEST - REALISTIC 15min (Q1 2026)")
    logger.info(f"{'='*70}\n")

    model, seq_len, feat_dim = load_model()
    if model is None:
        return

    # Load features
    df = pd.read_csv(DATA_DIR / f'{CRYPTO.lower()}_multi_tf_merged.csv')
    df['date'] = pd.to_datetime(df['date'])

    with open(MODEL_DIR / f'{CRYPTO.lower()}_cnn_features.json') as f:
        feature_cols = json.load(f)

    scaler = joblib.load(MODEL_DIR / 'feature_scaler.joblib')

    # Prepare features
    df_wide = df[df['date'] >= '2025-01-01'].copy()
    all_feat = df_wide[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    all_feat = np.clip(np.nan_to_num(scaler.transform(all_feat), nan=0, posinf=0, neginf=0), -5, 5)

    df_test = df_wide[(df_wide['date'] >= BACKTEST_START) & (df_wide['date'] <= BACKTEST_END)].copy()
    test_start_idx = len(df_wide) - len(df_test)

    logger.info(f"Test: {len(df_test)} days | Features: {len(feature_cols)} | Seq: {seq_len}")

    # 15min candles
    df_15m = fetch_15min_data()

    capital = 1000.0
    trades = []
    consec_losses = 0
    cooldown_until = None
    total_signals = 0
    filtered = 0
    filter_reasons = {}

    logger.info(f"Conf > {CONFIDENCE_THRESHOLD} | Dynamic TP/SL: {USE_DYNAMIC_TP_SL}\n")

    for i in range(len(df_test)):
        wide_idx = test_start_idx + i
        if wide_idx < seq_len:
            continue

        row = df_test.iloc[i]
        date = row['date']

        seq = all_feat[wide_idx-seq_len:wide_idx]
        X = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            direction, confidence = model.predict_direction(X)

        d, c = direction.item(), confidence.item()
        if d != 1 or c <= CONFIDENCE_THRESHOLD:
            continue

        total_signals += 1

        passes, reason = check_filters(row, consec_losses, cooldown_until, date)
        if not passes:
            filtered += 1
            filter_reasons[reason] = filter_reasons.get(reason, 0) + 1
            continue

        entry_price = row['close'] * (1 + SLIPPAGE)
        tp_price, sl_price = get_dynamic_tp_sl(row, entry_price)

        pos_val = capital * POSITION_SIZE
        shares = pos_val / entry_price
        capital -= pos_val + pos_val * TRADING_FEE

        result = simulate_trade_15min(entry_price, date, df_15m, tp_price, sl_price)

        exit_val = shares * result['exit_price']
        capital += exit_val - exit_val * TRADING_FEE
        pnl_pct = (result['exit_price'] / entry_price - 1) * 100

        trades.append({
            'entry_date': date, 'entry_price': entry_price, 'confidence': c,
            'exit_type': result['exit_type'], 'exit_price': result['exit_price'],
            'pnl_pct': pnl_pct, 'duration_h': result['duration_min'] / 60
        })

        if result['exit_type'] == 'SL':
            consec_losses += 1
            if consec_losses >= MAX_CONSECUTIVE_LOSSES:
                cooldown_until = date + timedelta(days=COOLDOWN_DAYS)
                logger.info(f"  >>> {consec_losses} losses -> cooldown until {cooldown_until.date()}")
        else:
            consec_losses = 0

        logger.info(f"  {date.date()} | Conf:{c:.1%} | {result['exit_type']} @ ${result['exit_price']:.2f} | "
                    f"PnL:{pnl_pct:+.2f}% | {result['duration_min']/60:.0f}h | ${capital:.0f}")

    # Results
    logger.info(f"\n{'='*70}")
    logger.info(f"RESULTS")
    logger.info(f"{'='*70}\n")

    logger.info(f"Signals: {total_signals} | Filtered: {filtered} | Executed: {len(trades)}")
    for r, cnt in sorted(filter_reasons.items(), key=lambda x: -x[1]):
        logger.info(f"  {r}: {cnt}")

    if trades:
        tdf = pd.DataFrame(trades)
        ret = (capital / 1000 - 1) * 100
        tp = len(tdf[tdf['exit_type'] == 'TP'])
        sl = len(tdf[tdf['exit_type'] == 'SL'])
        eod = len(tdf[tdf['exit_type'] == 'EOD'])
        wr = tp / len(trades) * 100

        logger.info(f"\n  Trades: {len(trades)} | TP:{tp} SL:{sl} EOD:{eod}")
        logger.info(f"  Win Rate: {wr:.1f}%")
        logger.info(f"  Return: {ret:+.2f}%")
        logger.info(f"  Capital: ${capital:.2f}")

        tdf.to_csv(RESULTS_DIR / f'{CRYPTO.lower()}_cnn_backtest.csv', index=False)
        with open(RESULTS_DIR / f'{CRYPTO.lower()}_cnn_summary.json', 'w') as f:
            json.dump({'trades': len(trades), 'tp': tp, 'sl': sl, 'wr': round(wr, 1),
                       'return': round(ret, 2), 'capital': round(capital, 2)}, f, indent=2, default=str)
    else:
        logger.info("\nNo trades")


if __name__ == "__main__":
    backtest()
