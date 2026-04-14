"""
BTC Independent LONG + SHORT Backtest
=======================================
Both models trade independently with their own filters.
Only rule: no LONG and SHORT at the same time on same coin.

Usage:
    python 04_backtest_independent.py
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

# Import DeepCNNShortModel from training script
sys.path.insert(0, str(BASE_DIR))
from importlib import import_module

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = BASE_DIR / 'data' / 'cache'
LONG_MODEL_DIR = BASE_DIR / 'models'
SHORT_MODEL_DIR = BASE_DIR / 'models_short'
RESULTS_DIR = BASE_DIR / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Trading params
POSITION_SIZE = 0.95
TRADING_FEE = 0.001
SLIPPAGE = 0.0005
USE_DYNAMIC_TP_SL = True
TP_PCT = 0.015
SL_PCT = 0.0075

# Thresholds (per direction)
LONG_CONF = 0.85   # Very selective - only strongest LONG signals
SHORT_CONF = 0.55  # SHORT performs well on LINK

# Filters
MAX_CONSEC_LOSSES = 2
COOLDOWN_DAYS = 2

BACKTEST_START = '2026-01-01'
BACKTEST_END = '2026-04-01'


def load_model(model_dir, model_file):
    path = model_dir / model_file
    if not path.exists():
        return None, None, None
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    feat_dim = ckpt.get('feature_dim', 99)
    seq_len = ckpt.get('sequence_length', 30)
    model_type = ckpt.get('model_type', 'cnn')

    # Auto-detect architecture from state_dict keys (same as signal_generator.py)
    keys = ckpt['model_state_dict'].keys()
    is_deep = any('conv3_1' in k or 'conv9_1' in k for k in keys)
    is_ln = any('ln1.weight' in k for k in keys)
    if is_deep:
        spec = import_module('03_train_short_model')
        model = spec.DeepCNNShortModel(feature_dim=feat_dim, sequence_length=seq_len, dropout=0.35)
    elif is_ln:
        # LINK uses DeepCNNShortModel with LayerNorm (no deep conv keys)
        spec = import_module('03_train_short_model')
        model = spec.DeepCNNShortModel(feature_dim=feat_dim, sequence_length=seq_len, dropout=0.30)
    else:
        model = CNNDirectionModel(feature_dim=feat_dim, sequence_length=seq_len, dropout=0.4)

    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, seq_len, ckpt


def fetch_15min():
    # Use pre-downloaded 15m data
    full = DATA_DIR / 'link_15m_data.csv'
    if full.exists():
        df = pd.read_csv(full)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df = df[(df.index >= BACKTEST_START) & (df.index <= BACKTEST_END)]
        logger.info(f"Loaded {len(df)} 15min candles for backtest")
        return df
    # Fallback: download from API
    ex = ccxt.binance({'enableRateLimit': True})
    s = int(pd.Timestamp(BACKTEST_START).timestamp() * 1000)
    e = int(pd.Timestamp(BACKTEST_END).timestamp() * 1000)
    candles = []
    cur = s
    while cur < e:
        c = ex.fetch_ohlcv('LINK/USDT', '15m', since=cur, limit=1000)
        if not c: break
        candles.extend(c)
        cur = c[-1][0] + 1
    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df


def get_tp_sl(row, entry, direction, s_tp=0.020, s_sl=0.010):
    """ATR-based symmetric TP/SL for V3 (matches live bot signal_generator.py)"""
    atr = row.get('1d_atr_14', None)
    ATR_MULT = 1.5
    if USE_DYNAMIC_TP_SL and atr and pd.notna(atr) and atr > 0:
        tp_m = min(max(ATR_MULT * atr / entry, 0.008), 0.04)
        sl_m = tp_m  # Symmetric
    else:
        tp_m, sl_m = 0.012, 0.012
    if direction == 'LONG':
        return entry * (1 + tp_m), entry * (1 - sl_m)
    else:
        return entry * (1 - tp_m), entry * (1 + sl_m)


def sim_trade(entry, date, df_15m, tp, sl, direction, max_d=10):
    for d in range(1, max_d + 1):
        day = date + timedelta(days=d)
        intra = df_15m[(df_15m.index >= day) & (df_15m.index < day + timedelta(days=1))]
        for ts, c in intra.iterrows():
            if direction == 'LONG':
                if c['high'] >= tp:
                    return {'type': 'TP', 'price': tp * (1 - SLIPPAGE), 'time': ts}
                if c['low'] <= sl:
                    return {'type': 'SL', 'price': sl * (1 - SLIPPAGE), 'time': ts}
            else:
                if c['low'] <= tp:
                    return {'type': 'TP', 'price': tp * (1 + SLIPPAGE), 'time': ts}
                if c['high'] >= sl:
                    return {'type': 'SL', 'price': sl * (1 + SLIPPAGE), 'time': ts}
    rem = df_15m[df_15m.index <= date + timedelta(days=max_d)]
    fp = rem.iloc[-1]['close'] if len(rem) > 0 else entry
    return {'type': 'EOD', 'price': fp, 'time': date + timedelta(days=max_d)}


def check_long_filters(row, consec, cool, date):
    """LONG-specific filters (same as 04_backtest.py)"""
    if cool and date <= cool:
        return False, "cooldown"
    # Momentum
    bull = sum(1 for c in ['1d_momentum_5', '4h_momentum_5', '1w_momentum_5']
               if c in row.index and pd.notna(row[c]) and row[c] > 0)
    total = sum(1 for c in ['1d_momentum_5', '4h_momentum_5', '1w_momentum_5']
                if c in row.index and pd.notna(row[c]))
    if total > 0 and bull < 1:
        return False, "weak_momentum"
    # Bear market
    if 'distance_from_sma50' in row.index and pd.notna(row['distance_from_sma50']):
        if row['distance_from_sma50'] < -0.05:
            return False, "bear_sma50"
    if 'distance_from_sma20' in row.index and pd.notna(row['distance_from_sma20']):
        if row['distance_from_sma20'] < -0.02:
            return False, "bear_sma20"
    # Weekly momentum (matches live bot)
    if '1w_momentum_5' in row.index and pd.notna(row['1w_momentum_5']):
        if row['1w_momentum_5'] < -0.10:
            return False, "weak_weekly"
    # Volatility
    if 'volatility_regime' in row.index and pd.notna(row['volatility_regime']):
        if row['volatility_regime'] > 2.5:
            return False, "high_vol"
    # Trend
    if 'trend_score' in row.index and pd.notna(row['trend_score']):
        if row['trend_score'] < -3:
            return False, "downtrend"
    return True, "pass"


def check_short_filters(row, consec, cool, date):
    """SHORT-specific filters (inverted logic)"""
    if cool and date <= cool:
        return False, "cooldown"
    # Bearish momentum required (inverted)
    bear = sum(1 for c in ['1d_momentum_5', '4h_momentum_5', '1w_momentum_5']
               if c in row.index and pd.notna(row[c]) and row[c] < 0)
    total = sum(1 for c in ['1d_momentum_5', '4h_momentum_5', '1w_momentum_5']
                if c in row.index and pd.notna(row[c]))
    if total > 0 and bear < 1:
        return False, "weak_bear_momentum"
    # Don't short in bull market
    if 'distance_from_sma50' in row.index and pd.notna(row['distance_from_sma50']):
        if row['distance_from_sma50'] > 0.05:
            return False, "bull_sma50"
    if 'distance_from_sma20' in row.index and pd.notna(row['distance_from_sma20']):
        if row['distance_from_sma20'] > 0.03:
            return False, "bull_sma20"
    # Volatility
    if 'volatility_regime' in row.index and pd.notna(row['volatility_regime']):
        if row['volatility_regime'] > 2.5:
            return False, "high_vol"
    # Uptrend = bad for short
    if 'trend_score' in row.index and pd.notna(row['trend_score']):
        if row['trend_score'] > 3:
            return False, "uptrend"
    return True, "pass"


def backtest():
    logger.info(f"\n{'='*70}")
    logger.info(f"LINK INDEPENDENT LONG + SHORT BACKTEST (Q1 2026)")
    logger.info(f"{'='*70}\n")

    # Load models
    long_model, long_seq, long_ckpt = load_model(LONG_MODEL_DIR, 'LINK_direction_model.pt')
    short_model, short_seq, short_ckpt = load_model(SHORT_MODEL_DIR, 'LINK_short_model.pt')
    if not long_model or not short_model:
        logger.error("Missing model(s)")
        return

    # LONG features
    with open(BASE_DIR / 'required_features.json') as f:
        long_feature_cols = json.load(f)

    # SHORT features (may include bear-specific features)
    with open(SHORT_MODEL_DIR / 'short_features.json') as f:
        short_feature_cols = json.load(f)

    long_scaler = joblib.load(LONG_MODEL_DIR / 'feature_scaler.joblib')
    short_scaler = joblib.load(SHORT_MODEL_DIR / 'feature_scaler_short.joblib')

    # Get SHORT TP/SL from checkpoint
    short_tp = short_ckpt.get('short_tp_pct', 0.020) if short_ckpt else 0.020
    short_sl = short_ckpt.get('short_sl_pct', 0.010) if short_ckpt else 0.010
    long_temp = long_ckpt.get('temperature', 1.0) if long_ckpt else 1.0
    short_temp = short_ckpt.get('temperature', 1.0) if short_ckpt else 1.0
    logger.info(f"SHORT params: TP={short_tp:.1%} drop, SL={short_sl:.1%} rise")
    logger.info(f"Temperature: LONG={long_temp:.3f}, SHORT={short_temp:.3f}")

    df = pd.read_csv(DATA_DIR / 'link_features.csv')
    df['date'] = pd.to_datetime(df['date'])

    # Add bear features for SHORT model
    from importlib import import_module as imp
    try:
        short_train = imp('03_train_short_model')
        df = short_train.add_bear_features(df)
    except:
        pass

    df_wide = df[df['date'] >= '2025-01-01'].copy()

    # Prepare LONG features
    long_raw = df_wide[long_feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    long_feat = np.clip(np.nan_to_num(long_scaler.transform(long_raw), nan=0, posinf=0, neginf=0), -5, 5)

    # Prepare SHORT features (different columns)
    for c in short_feature_cols:
        if c not in df_wide.columns:
            df_wide[c] = 0
    short_raw = df_wide[short_feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    short_feat = np.clip(np.nan_to_num(short_scaler.transform(short_raw), nan=0, posinf=0, neginf=0), -5, 5)

    df_test = df_wide[(df_wide['date'] >= BACKTEST_START) & (df_wide['date'] <= BACKTEST_END)].copy()
    t_start = len(df_wide) - len(df_test)
    seq = max(long_seq, short_seq)

    df_15m = fetch_15min()

    logger.info(f"Test: {len(df_test)} days | 15min: {len(df_15m)} candles")
    logger.info(f"LONG conf >= {LONG_CONF} | SHORT conf >= {SHORT_CONF}\n")

    capital = 1000.0
    trades = []
    position = None  # None, 'LONG', or 'SHORT'
    long_consec = 0
    short_consec = 0
    long_cool = None
    short_cool = None
    long_filtered = {}
    short_filtered = {}

    for i in range(len(df_test)):
        wi = t_start + i
        if wi < seq:
            continue

        row = df_test.iloc[i]
        date = row['date']

        if position is not None:
            continue  # Already in a trade

        # LONG prediction (with temperature-calibrated confidence)
        lx = torch.tensor(long_feat[wi-long_seq:wi], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = long_model(lx)
            probs = torch.softmax(logits / long_temp, dim=1)
            l_c, l_d = torch.max(probs, dim=1)
        l_d, l_c = l_d.item(), l_c.item()

        # SHORT prediction (with temperature-calibrated confidence)
        sx = torch.tensor(short_feat[wi-short_seq:wi], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = short_model(sx)
            probs = torch.softmax(logits / short_temp, dim=1)
            s_c, s_d = torch.max(probs, dim=1)
        s_d, s_c = s_d.item(), s_c.item()

        # Try LONG first (priority)
        if l_d == 1 and l_c >= LONG_CONF:
            ok, reason = check_long_filters(row, long_consec, long_cool, date)
            if ok:
                entry = row['close'] * (1 + SLIPPAGE)
                tp, sl = get_tp_sl(row, entry, 'LONG')
                pos_val = capital * POSITION_SIZE
                shares = pos_val / entry
                capital -= pos_val + pos_val * TRADING_FEE

                res = sim_trade(entry, date, df_15m, tp, sl, 'LONG')
                exit_val = shares * res['price']
                capital += exit_val - exit_val * TRADING_FEE
                pnl = (res['price'] / entry - 1) * 100

                trades.append({'date': date, 'dir': 'LONG', 'entry': entry, 'conf': l_c,
                               'exit_type': res['type'], 'exit_price': res['price'],
                               'pnl': pnl, 'dur_h': (res['time'] - date).total_seconds() / 3600})

                if res['type'] == 'SL':
                    long_consec += 1
                    if long_consec >= MAX_CONSEC_LOSSES:
                        long_cool = date + timedelta(days=COOLDOWN_DAYS)
                else:
                    long_consec = 0

                logger.info(f"  {date.date()} | LONG  | Conf:{l_c:.0%} | {res['type']} | PnL:{pnl:+.2f}% | ${capital:.0f}")
                continue
            else:
                long_filtered[reason] = long_filtered.get(reason, 0) + 1

        # Try SHORT if LONG didn't trigger
        if s_d == 1 and s_c >= SHORT_CONF:
            ok, reason = check_short_filters(row, short_consec, short_cool, date)
            if ok:
                entry = row['close'] * (1 - SLIPPAGE)
                tp, sl = get_tp_sl(row, entry, 'SHORT', short_tp, short_sl)
                pos_val = capital * POSITION_SIZE
                capital -= pos_val * TRADING_FEE

                res = sim_trade(entry, date, df_15m, tp, sl, 'SHORT')
                pnl = (entry / res['price'] - 1) * 100
                exit_val = pos_val * (1 + pnl / 100)
                capital = capital - pos_val + exit_val - exit_val * TRADING_FEE

                trades.append({'date': date, 'dir': 'SHORT', 'entry': entry, 'conf': s_c,
                               'exit_type': res['type'], 'exit_price': res['price'],
                               'pnl': pnl, 'dur_h': (res['time'] - date).total_seconds() / 3600})

                if res['type'] == 'SL':
                    short_consec += 1
                    if short_consec >= MAX_CONSEC_LOSSES:
                        short_cool = date + timedelta(days=COOLDOWN_DAYS)
                else:
                    short_consec = 0

                logger.info(f"  {date.date()} | SHORT | Conf:{s_c:.0%} | {res['type']} | PnL:{pnl:+.2f}% | ${capital:.0f}")
                continue
            else:
                short_filtered[reason] = short_filtered.get(reason, 0) + 1

    # Results
    logger.info(f"\n{'='*70}")
    logger.info(f"RESULTS - INDEPENDENT LONG + SHORT")
    logger.info(f"{'='*70}\n")

    if long_filtered:
        logger.info("LONG filtered:")
        for r, c in sorted(long_filtered.items(), key=lambda x: -x[1]):
            logger.info(f"  {r}: {c}")
    if short_filtered:
        logger.info("SHORT filtered:")
        for r, c in sorted(short_filtered.items(), key=lambda x: -x[1]):
            logger.info(f"  {r}: {c}")

    if trades:
        tdf = pd.DataFrame(trades)
        ret = (capital / 1000 - 1) * 100

        lt = tdf[tdf['dir'] == 'LONG']
        st = tdf[tdf['dir'] == 'SHORT']
        lt_tp = len(lt[lt['exit_type'] == 'TP'])
        lt_sl = len(lt[lt['exit_type'] == 'SL'])
        st_tp = len(st[st['exit_type'] == 'TP'])
        st_sl = len(st[st['exit_type'] == 'SL'])
        total_tp = lt_tp + st_tp
        total_sl = lt_sl + st_sl
        wr = total_tp / len(trades) * 100

        logger.info(f"\n  LONG:  {len(lt)} trades | TP:{lt_tp} SL:{lt_sl} | WR:{lt_tp/len(lt)*100:.0f}%" if len(lt) > 0 else "")
        logger.info(f"  SHORT: {len(st)} trades | TP:{st_tp} SL:{st_sl} | WR:{st_tp/len(st)*100:.0f}%" if len(st) > 0 else "")
        logger.info(f"\n  TOTAL: {len(trades)} trades | TP:{total_tp} SL:{total_sl} | WR: {wr:.1f}%")
        logger.info(f"  Return: {ret:+.2f}%")
        logger.info(f"  Capital: $1000 -> ${capital:.2f}")

        tdf.to_csv(RESULTS_DIR / 'link_independent_backtest.csv', index=False)
    else:
        logger.info("No trades")


if __name__ == "__main__":
    backtest()
