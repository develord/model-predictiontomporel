"""
BTC Combined LONG+SHORT Backtest with Voting System
=====================================================
Uses both LONG and SHORT CNN models to make trading decisions.

Voting logic:
  - LONG model predicts BUY + SHORT model predicts "no short" → OPEN LONG
  - SHORT model predicts SELL + LONG model predicts "no long" → OPEN SHORT
  - Both agree or both uncertain → NO TRADE

Usage:
    python 04_backtest_combined.py
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
LONG_MODEL_DIR = BASE_DIR / 'models'
SHORT_MODEL_DIR = BASE_DIR / 'models_short'
RESULTS_DIR = BASE_DIR / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Trading params
TP_PCT = 0.015
SL_PCT = 0.0075
POSITION_SIZE = 0.95
TRADING_FEE = 0.001
SLIPPAGE = 0.0005
USE_DYNAMIC_TP_SL = True

# Confidence thresholds
LONG_CONF_THRESHOLD = 0.60
SHORT_CONF_THRESHOLD = 0.60

# Filters
MAX_CONSECUTIVE_LOSSES = 2
COOLDOWN_DAYS = 5

BACKTEST_START = '2026-01-01'
BACKTEST_END = '2026-03-24'


def load_model(model_dir, model_file):
    """Load a CNN model"""
    path = model_dir / model_file
    if not path.exists():
        logger.error(f"Model not found: {path}")
        return None, None
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    feat_dim = ckpt.get('feature_dim', 99)
    seq_len = ckpt.get('sequence_length', 30)
    model = CNNDirectionModel(feature_dim=feat_dim, sequence_length=seq_len, dropout=0.4)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, seq_len


def fetch_15min():
    """Fetch 15min candles"""
    cache = DATA_DIR / 'btc_15min_q1_2026.csv'
    if cache.exists():
        df = pd.read_csv(cache, parse_dates=['timestamp'])
        df.set_index('timestamp', inplace=True)
        return df
    logger.info("Downloading 15min candles...")
    ex = ccxt.binance({'enableRateLimit': True})
    start_ts = int(pd.Timestamp(BACKTEST_START).timestamp() * 1000)
    end_ts = int(pd.Timestamp(BACKTEST_END).timestamp() * 1000)
    all_c = []
    cur = start_ts
    while cur < end_ts:
        c = ex.fetch_ohlcv('BTC/USDT', '15m', since=cur, limit=1000)
        if not c: break
        all_c.extend(c)
        cur = c[-1][0] + 1
    df = pd.DataFrame(all_c, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[(df.index >= BACKTEST_START) & (df.index < BACKTEST_END)]
    df.to_csv(cache)
    return df


def get_dynamic_tp_sl(row, entry_price, direction='long'):
    """Dynamic TP/SL based on ATR"""
    atr = row.get('1d_atr_14', None)
    if USE_DYNAMIC_TP_SL and atr is not None and pd.notna(atr) and atr > 0:
        tp_mult = min(max(atr / entry_price, 0.008), 0.03)
        sl_mult = tp_mult * 0.5
    else:
        tp_mult, sl_mult = TP_PCT, SL_PCT

    if direction == 'long':
        return entry_price * (1 + tp_mult), entry_price * (1 - sl_mult)
    else:  # short
        return entry_price * (1 - tp_mult), entry_price * (1 + sl_mult)


def simulate_trade_15min(entry_price, entry_date, df_15m, tp_price, sl_price, direction='long', max_days=10):
    """Simulate trade with 15min candles"""
    for d in range(1, max_days + 1):
        day = entry_date + timedelta(days=d)
        intra = df_15m[(df_15m.index >= day) & (df_15m.index < day + timedelta(days=1))]
        for ts, c in intra.iterrows():
            if direction == 'long':
                if c['high'] >= tp_price:
                    return {'type': 'TP', 'price': tp_price * (1 - SLIPPAGE), 'time': ts}
                if c['low'] <= sl_price:
                    return {'type': 'SL', 'price': sl_price * (1 - SLIPPAGE), 'time': ts}
            else:  # short
                if c['low'] <= tp_price:  # TP when price drops
                    return {'type': 'TP', 'price': tp_price * (1 + SLIPPAGE), 'time': ts}
                if c['high'] >= sl_price:  # SL when price rises
                    return {'type': 'SL', 'price': sl_price * (1 + SLIPPAGE), 'time': ts}

    rem = df_15m[df_15m.index <= entry_date + timedelta(days=max_days)]
    fp = rem.iloc[-1]['close'] if len(rem) > 0 else entry_price
    return {'type': 'EOD', 'price': fp, 'time': entry_date + timedelta(days=max_days)}


def check_filters(row, consec_losses, cooldown_until, date):
    """Signal filters"""
    if cooldown_until and date <= cooldown_until:
        return False, "cooldown"
    if 'volatility_regime' in row.index and pd.notna(row['volatility_regime']):
        if row['volatility_regime'] > 2.5:
            return False, "high_vol"
    if '1d_adx_14' in row.index and pd.notna(row['1d_adx_14']):
        if row['1d_adx_14'] < 15:
            return False, "no_trend"
    return True, "pass"


def check_long_filters(row):
    """Extra filters for LONG only"""
    if 'distance_from_sma50' in row.index and pd.notna(row['distance_from_sma50']):
        if row['distance_from_sma50'] < -0.05:
            return False, "bear_sma50"
    if 'distance_from_sma20' in row.index and pd.notna(row['distance_from_sma20']):
        if row['distance_from_sma20'] < -0.02:
            return False, "bear_sma20"
    if 'trend_score' in row.index and pd.notna(row['trend_score']):
        if row['trend_score'] < -3:
            return False, "downtrend"
    return True, "pass"


def check_short_filters(row):
    """Extra filters for SHORT only"""
    # Don't short in strong bull market
    if 'distance_from_sma50' in row.index and pd.notna(row['distance_from_sma50']):
        if row['distance_from_sma50'] > 0.10:
            return False, "bull_sma50"
    if 'distance_from_sma20' in row.index and pd.notna(row['distance_from_sma20']):
        if row['distance_from_sma20'] > 0.05:
            return False, "bull_sma20"
    # Don't short when trend score is very positive
    if 'trend_score' in row.index and pd.notna(row['trend_score']):
        if row['trend_score'] > 3:
            return False, "uptrend"
    return True, "pass"


def backtest():
    logger.info(f"\n{'='*70}")
    logger.info(f"BTC COMBINED LONG+SHORT BACKTEST (Q1 2026)")
    logger.info(f"{'='*70}\n")

    # Load both models
    long_model, long_seq = load_model(LONG_MODEL_DIR, 'BTC_direction_model.pt')
    short_model, short_seq = load_model(SHORT_MODEL_DIR, 'BTC_short_model.pt')

    if long_model is None or short_model is None:
        logger.error("Missing models")
        return

    logger.info(f"LONG model loaded (seq={long_seq})")
    logger.info(f"SHORT model loaded (seq={short_seq})")

    # Load features + scalers
    with open(BASE_DIR / 'required_features.json') as f:
        feature_cols = json.load(f)

    long_scaler = joblib.load(LONG_MODEL_DIR / 'feature_scaler.joblib')
    short_scaler = joblib.load(SHORT_MODEL_DIR / 'feature_scaler_short.joblib')

    # Load data
    df = pd.read_csv(DATA_DIR / 'btc_features.csv')
    df['date'] = pd.to_datetime(df['date'])

    df_wide = df[df['date'] >= '2025-01-01'].copy()
    feat_raw = df_wide[feature_cols].fillna(0).values.astype(np.float32)

    long_feat = np.clip(np.nan_to_num(long_scaler.transform(feat_raw), nan=0, posinf=0, neginf=0), -5, 5)
    short_feat = np.clip(np.nan_to_num(short_scaler.transform(feat_raw), nan=0, posinf=0, neginf=0), -5, 5)

    df_test = df_wide[(df_wide['date'] >= BACKTEST_START) & (df_wide['date'] <= BACKTEST_END)].copy()
    t_start = len(df_wide) - len(df_test)

    logger.info(f"Test: {len(df_test)} days | Features: {len(feature_cols)}")

    # 15min candles
    df_15m = fetch_15min()
    logger.info(f"15min: {len(df_15m)} candles\n")

    seq_len = max(long_seq, short_seq)

    # Trading simulation
    capital = 1000.0
    trades = []
    in_position = False
    consec_losses = 0
    cooldown_until = None
    filter_reasons = {}

    for i in range(len(df_test)):
        wi = t_start + i
        if wi < seq_len:
            continue

        row = df_test.iloc[i]
        date = row['date']

        if in_position:
            continue

        # Get predictions from both models
        long_X = torch.tensor(long_feat[wi-long_seq:wi], dtype=torch.float32).unsqueeze(0)
        short_X = torch.tensor(short_feat[wi-short_seq:wi], dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            long_dir, long_conf = long_model.predict_direction(long_X)
            short_dir, short_conf = short_model.predict_direction(short_X)

        l_d, l_c = long_dir.item(), long_conf.item()
        s_d, s_c = short_dir.item(), short_conf.item()

        # Voting logic
        signal = None
        if l_d == 1 and l_c >= LONG_CONF_THRESHOLD and s_d == 0:
            # LONG model says BUY, SHORT model says "not a short opportunity"
            signal = 'LONG'
            conf = l_c
        elif s_d == 1 and s_c >= SHORT_CONF_THRESHOLD and l_d == 0:
            # SHORT model says SELL, LONG model says "not a buy opportunity"
            signal = 'SHORT'
            conf = s_c
        else:
            continue  # No consensus

        # Common filters
        ok, reason = check_filters(row, consec_losses, cooldown_until, date)
        if not ok:
            filter_reasons[reason] = filter_reasons.get(reason, 0) + 1
            continue

        # Direction-specific filters
        if signal == 'LONG':
            ok, reason = check_long_filters(row)
        else:
            ok, reason = check_short_filters(row)
        if not ok:
            filter_reasons[reason] = filter_reasons.get(reason, 0) + 1
            continue

        # Execute trade
        entry_price = row['close'] * (1 + SLIPPAGE) if signal == 'LONG' else row['close'] * (1 - SLIPPAGE)
        tp_price, sl_price = get_dynamic_tp_sl(row, entry_price, signal.lower())

        pos_val = capital * POSITION_SIZE
        if signal == 'LONG':
            shares = pos_val / entry_price
        else:
            shares = pos_val / entry_price  # Short sell quantity

        capital -= pos_val * TRADING_FEE  # Fee on entry

        result = simulate_trade_15min(entry_price, date, df_15m, tp_price, sl_price, signal.lower())

        # Calculate P&L
        if signal == 'LONG':
            pnl_pct = (result['price'] / entry_price - 1) * 100
            exit_val = shares * result['price']
        else:
            pnl_pct = (entry_price / result['price'] - 1) * 100  # Profit from price drop
            exit_val = pos_val + (pos_val * pnl_pct / 100)

        capital = capital - pos_val + exit_val - (exit_val * TRADING_FEE)

        dur_h = (result['time'] - date).total_seconds() / 3600

        trades.append({
            'date': date, 'signal': signal, 'entry': entry_price, 'conf': conf,
            'exit_type': result['type'], 'exit_price': result['price'],
            'pnl_pct': pnl_pct, 'duration_h': dur_h,
            'long_pred': f"{'BUY' if l_d==1 else 'NOBUY'} {l_c:.0%}",
            'short_pred': f"{'SELL' if s_d==1 else 'NOSELL'} {s_c:.0%}",
        })

        # Track losses
        if result['type'] == 'SL':
            consec_losses += 1
            if consec_losses >= MAX_CONSECUTIVE_LOSSES:
                cooldown_until = date + timedelta(days=COOLDOWN_DAYS)
        else:
            consec_losses = 0

        logger.info(f"  {date.date()} | {signal:5s} | Conf:{conf:.0%} | "
                    f"{result['type']} | PnL:{pnl_pct:+.2f}% | {dur_h:.0f}h | "
                    f"L:{l_d}({l_c:.0%}) S:{s_d}({s_c:.0%}) | ${capital:.0f}")

    # Results
    logger.info(f"\n{'='*70}")
    logger.info(f"COMBINED BACKTEST RESULTS")
    logger.info(f"{'='*70}\n")

    if filter_reasons:
        logger.info("Filtered:")
        for r, cnt in sorted(filter_reasons.items(), key=lambda x: -x[1]):
            logger.info(f"  {r}: {cnt}")

    if trades:
        tdf = pd.DataFrame(trades)
        ret = (capital / 1000 - 1) * 100

        long_trades = tdf[tdf['signal'] == 'LONG']
        short_trades = tdf[tdf['signal'] == 'SHORT']

        long_tp = len(long_trades[long_trades['exit_type'] == 'TP'])
        long_sl = len(long_trades[long_trades['exit_type'] == 'SL'])
        short_tp = len(short_trades[short_trades['exit_type'] == 'TP'])
        short_sl = len(short_trades[short_trades['exit_type'] == 'SL'])

        total_tp = long_tp + short_tp
        total_sl = long_sl + short_sl
        wr = total_tp / len(trades) * 100 if trades else 0

        logger.info(f"\nTotal Trades: {len(trades)} | TP: {total_tp} | SL: {total_sl} | WR: {wr:.1f}%")
        logger.info(f"  LONG:  {len(long_trades)} trades (TP:{long_tp} SL:{long_sl})")
        logger.info(f"  SHORT: {len(short_trades)} trades (TP:{short_tp} SL:{short_sl})")
        logger.info(f"\nReturn: {ret:+.2f}%")
        logger.info(f"Capital: $1000 -> ${capital:.2f}")

        tdf.to_csv(RESULTS_DIR / 'btc_combined_backtest.csv', index=False)
    else:
        logger.info("No trades executed")


if __name__ == "__main__":
    backtest()
