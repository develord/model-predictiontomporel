"""
ETH Meta-Model Backtest
========================
CNN LONG/SHORT predict -> XGBoost meta decides -> Execute trade

Compares: CNN-only vs CNN+Meta side by side

Usage:
    python 06_backtest_meta.py
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
META_MODEL_DIR = BASE_DIR / 'models_meta'
RESULTS_DIR = BASE_DIR / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

POSITION_SIZE = 0.95
TRADING_FEE = 0.001
SLIPPAGE = 0.0005
USE_DYNAMIC_TP_SL = True
TP_PCT = 0.015
SL_PCT = 0.0075

# CNN thresholds (kept as minimum gate)
LONG_CNN_CONF = 0.55   # Lowered: meta-model does the real filtering
SHORT_CNN_CONF = 0.50

# Meta thresholds (this is the real filter now)
LONG_META_CONF = 0.50
SHORT_META_CONF = 0.40  # Lower: meta training data was bull-biased, let more shorts through

MAX_CONSEC_LOSSES = 3
COOLDOWN_DAYS = 3

BACKTEST_START = '2026-01-01'
BACKTEST_END = '2026-04-05'


def load_cnn(model_dir, model_file):
    path = model_dir / model_file
    if not path.exists():
        return None, None, None
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    model = CNNDirectionModel(
        feature_dim=ckpt.get('feature_dim', 99),
        sequence_length=ckpt.get('sequence_length', 30),
        dropout=0.4
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, ckpt.get('sequence_length', 30), ckpt


def fetch_15min():
    cache = DATA_DIR / 'eth_15min_q1_2026.csv'
    if cache.exists():
        df = pd.read_csv(cache, parse_dates=['timestamp'])
        df.set_index('timestamp', inplace=True)
        return df
    ex = ccxt.binance({'enableRateLimit': True})
    s = int(pd.Timestamp(BACKTEST_START).timestamp() * 1000)
    e = int(pd.Timestamp(BACKTEST_END).timestamp() * 1000)
    candles = []
    cur = s
    while cur < e:
        c = ex.fetch_ohlcv('ETH/USDT', '15m', since=cur, limit=1000)
        if not c: break
        candles.extend(c)
        cur = c[-1][0] + 1
    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[(df.index >= BACKTEST_START) & (df.index < BACKTEST_END)]
    df.to_csv(cache)
    return df


def get_tp_sl(row, entry, direction, s_tp=0.020, s_sl=0.010):
    atr = row.get('1d_atr_14', None)
    if direction == 'LONG':
        if USE_DYNAMIC_TP_SL and atr and pd.notna(atr) and atr > 0:
            tp_m = min(max(atr / entry, 0.008), 0.03)
            sl_m = tp_m * 0.5
        else:
            tp_m, sl_m = TP_PCT, SL_PCT
        return entry * (1 + tp_m), entry * (1 - sl_m)
    else:
        exec_tp = max(s_tp, 0.03)
        exec_sl = max(s_sl, 0.03)
        return entry * (1 - exec_tp), entry * (1 + exec_sl)


def sim_trade(entry, date, df_15m, tp, sl, direction, max_d=20):
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


def build_meta_features(row, long_conf, long_dir, short_conf, short_dir,
                        l_p0, l_p1, s_p0, s_p1):
    """Same as training script - must match exactly"""
    feat = {
        'long_conf': long_conf,
        'long_dir': long_dir,
        'short_conf': short_conf,
        'short_dir': short_dir,
        'long_prob_spread': l_p1 - l_p0,
        'short_prob_spread': s_p1 - s_p0,
        'models_agree_bull': int(long_dir == 1 and short_dir == 0),
        'models_agree_bear': int(long_dir == 0 and short_dir == 1),
        'models_conflict': int(long_dir == 1 and short_dir == 1),
        'models_neutral': int(long_dir == 0 and short_dir == 0),
        'conf_diff': long_conf - short_conf,
        'max_conf': max(long_conf, short_conf),
        'min_conf': min(long_conf, short_conf),
    }

    market_cols = [
        '1d_rsi_14', '1d_adx_14', '1d_atr_14', '1d_macd_histogram',
        '1d_bb_width', '1d_stoch_k', '1d_cmf_20',
        'volatility_regime', 'volume_trend', 'trend_score',
        'distance_from_sma20', 'distance_from_sma50',
        'price_position_20', 'price_position_50',
        'regime_bull', 'regime_bear', 'regime_range',
        'accumulation_score', 'distribution_score',
        'vwap_trend_10', 'pressure_ratio',
        'trend_consistency_10', 'trend_consistency_20',
        'resistance_dist_pct', 'support_dist_pct',
        'sma50_above_sma200', 'sma_spread_pct',
        'rsi_bullish_count', 'macd_bullish_count',
        'adx_mean', 'momentum_bullish_count',
        'consecutive_up', 'consecutive_down',
        'body_ratio', 'day_of_week',
    ]

    for col in market_cols:
        val = row.get(col, np.nan) if col in row.index else np.nan
        feat[col] = val if pd.notna(val) else 0.0

    return feat


def run_backtest(use_meta=True, label=""):
    """Run backtest with or without meta-model"""
    long_model, long_seq, long_ckpt = load_cnn(LONG_MODEL_DIR, 'ETH_direction_model.pt')
    short_model, short_seq, short_ckpt = load_cnn(SHORT_MODEL_DIR, 'ETH_short_model.pt')

    long_temp = long_ckpt.get('temperature', 1.0)
    short_temp = short_ckpt.get('temperature', 1.0)

    # Load meta-models if needed
    meta_long, meta_short, meta_feature_cols = None, None, None
    if use_meta:
        meta_long_path = META_MODEL_DIR / 'ETH_meta_long.joblib'
        meta_short_path = META_MODEL_DIR / 'ETH_meta_short.joblib'
        meta_feat_path = META_MODEL_DIR / 'meta_features.json'
        if meta_long_path.exists() and meta_short_path.exists():
            meta_long = joblib.load(meta_long_path)
            meta_short = joblib.load(meta_short_path)
            with open(meta_feat_path) as f:
                meta_feature_cols = json.load(f)
        else:
            logger.warning("Meta models not found, running without")
            use_meta = False

    with open(BASE_DIR / 'required_features.json') as f:
        long_feature_cols = json.load(f)
    with open(SHORT_MODEL_DIR / 'short_features.json') as f:
        short_feature_cols = json.load(f)

    long_scaler = joblib.load(LONG_MODEL_DIR / 'feature_scaler.joblib')
    short_scaler = joblib.load(SHORT_MODEL_DIR / 'feature_scaler_short.joblib')

    short_tp = short_ckpt.get('short_tp_pct', 0.020)
    short_sl = short_ckpt.get('short_sl_pct', 0.010)

    df = pd.read_csv(DATA_DIR / 'eth_features.csv')
    df['date'] = pd.to_datetime(df['date'])

    sys.path.insert(0, str(BASE_DIR))
    try:
        from importlib import import_module
        short_train = import_module('03_train_short_model')
        df = short_train.add_bear_features(df)
    except:
        pass

    df_wide = df[df['date'] >= '2025-01-01'].copy()

    long_raw = df_wide[long_feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    long_feat = np.clip(np.nan_to_num(long_scaler.transform(long_raw), nan=0, posinf=0, neginf=0), -5, 5)

    for c in short_feature_cols:
        if c not in df_wide.columns:
            df_wide[c] = 0
    short_raw = df_wide[short_feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    short_feat = np.clip(np.nan_to_num(short_scaler.transform(short_raw), nan=0, posinf=0, neginf=0), -5, 5)

    df_test = df_wide[(df_wide['date'] >= BACKTEST_START) & (df_wide['date'] <= BACKTEST_END)].copy()
    t_start = len(df_wide) - len(df_test)
    seq = max(long_seq, short_seq)

    df_15m = fetch_15min()

    capital = 1000.0
    trades = []
    long_consec, short_consec = 0, 0
    long_cool, short_cool = None, None

    for i in range(len(df_test)):
        wi = t_start + i
        if wi < seq:
            continue

        row = df_test.iloc[i]
        date = row['date']

        # CNN predictions
        lx = torch.tensor(long_feat[wi - long_seq:wi], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            l_logits = long_model(lx)
            l_probs = torch.softmax(l_logits / long_temp, dim=1).squeeze()
        l_conf, l_dir = l_probs.max(0)
        l_dir, l_conf = l_dir.item(), l_conf.item()
        l_p0, l_p1 = l_probs[0].item(), l_probs[1].item()

        sx = torch.tensor(short_feat[wi - short_seq:wi], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            s_logits = short_model(sx)
            s_probs = torch.softmax(s_logits / short_temp, dim=1).squeeze()
        s_conf, s_dir = s_probs.max(0)
        s_dir, s_conf = s_dir.item(), s_conf.item()
        s_p0, s_p1 = s_probs[0].item(), s_probs[1].item()

        # Build meta features
        mf = build_meta_features(row, l_conf, l_dir, s_conf, s_dir, l_p0, l_p1, s_p0, s_p1)

        # === TRY LONG ===
        if l_dir == 1 and l_conf >= LONG_CNN_CONF:
            # Cooldown check
            if long_cool and date <= long_cool:
                pass
            else:
                take_trade = True
                meta_prob = None

                if use_meta and meta_long is not None:
                    mf_vec = np.array([[mf.get(c, 0) for c in meta_feature_cols]])
                    meta_prob = meta_long.predict_proba(mf_vec)[0][1]
                    take_trade = meta_prob >= LONG_META_CONF

                if take_trade:
                    entry = row['close'] * (1 + SLIPPAGE)
                    tp, sl = get_tp_sl(row, entry, 'LONG')
                    pos_val = capital * POSITION_SIZE
                    shares = pos_val / entry
                    capital -= pos_val + pos_val * TRADING_FEE

                    res = sim_trade(entry, date, df_15m, tp, sl, 'LONG')
                    exit_val = shares * res['price']
                    capital += exit_val - exit_val * TRADING_FEE
                    pnl = (res['price'] / entry - 1) * 100

                    mp_str = f" Meta:{meta_prob:.0%}" if meta_prob is not None else ""
                    trades.append({
                        'date': date, 'dir': 'LONG', 'entry': entry,
                        'cnn_conf': l_conf, 'meta_prob': meta_prob,
                        'exit_type': res['type'], 'exit_price': res['price'],
                        'pnl': pnl, 'dur_h': (res['time'] - date).total_seconds() / 3600
                    })

                    if res['type'] == 'SL':
                        long_consec += 1
                        if long_consec >= MAX_CONSEC_LOSSES:
                            long_cool = date + timedelta(days=COOLDOWN_DAYS)
                    else:
                        long_consec = 0

                    logger.info(f"  {date.date()} | LONG  | CNN:{l_conf:.0%}{mp_str} | {res['type']} | PnL:{pnl:+.2f}% | ${capital:.0f}")
                    continue

        # === TRY SHORT ===
        if s_dir == 1 and s_conf >= SHORT_CNN_CONF:
            if short_cool and date <= short_cool:
                pass
            else:
                take_trade = True
                meta_prob = None

                if use_meta and meta_short is not None:
                    mf_vec = np.array([[mf.get(c, 0) for c in meta_feature_cols]])
                    meta_prob = meta_short.predict_proba(mf_vec)[0][1]
                    take_trade = meta_prob >= SHORT_META_CONF

                if take_trade:
                    entry = row['close'] * (1 - SLIPPAGE)
                    tp, sl = get_tp_sl(row, entry, 'SHORT', short_tp, short_sl)
                    pos_val = capital * POSITION_SIZE
                    capital -= pos_val * TRADING_FEE

                    res = sim_trade(entry, date, df_15m, tp, sl, 'SHORT')
                    pnl = (entry / res['price'] - 1) * 100
                    exit_val = pos_val * (1 + pnl / 100)
                    capital = capital - pos_val + exit_val - exit_val * TRADING_FEE

                    mp_str = f" Meta:{meta_prob:.0%}" if meta_prob is not None else ""
                    trades.append({
                        'date': date, 'dir': 'SHORT', 'entry': entry,
                        'cnn_conf': s_conf, 'meta_prob': meta_prob,
                        'exit_type': res['type'], 'exit_price': res['price'],
                        'pnl': pnl, 'dur_h': (res['time'] - date).total_seconds() / 3600
                    })

                    if res['type'] == 'SL':
                        short_consec += 1
                        if short_consec >= MAX_CONSEC_LOSSES:
                            short_cool = date + timedelta(days=COOLDOWN_DAYS)
                    else:
                        short_consec = 0

                    logger.info(f"  {date.date()} | SHORT | CNN:{s_conf:.0%}{mp_str} | {res['type']} | PnL:{pnl:+.2f}% | ${capital:.0f}")
                    continue

    return capital, trades, label


def backtest():
    logger.info(f"\n{'='*70}")
    logger.info(f"ETH META-MODEL BACKTEST COMPARISON")
    logger.info(f"{'='*70}\n")

    # Run with meta
    logger.info(f"--- CNN + META-MODEL ---")
    cap_meta, trades_meta, _ = run_backtest(use_meta=True, label="CNN+Meta")

    logger.info(f"\n--- CNN ONLY (no meta) ---")
    cap_cnn, trades_cnn, _ = run_backtest(use_meta=False, label="CNN-only")

    # Compare
    logger.info(f"\n{'='*70}")
    logger.info(f"COMPARISON")
    logger.info(f"{'='*70}\n")

    for label, capital, trades in [("CNN+Meta", cap_meta, trades_meta), ("CNN-only", cap_cnn, trades_cnn)]:
        if trades:
            tdf = pd.DataFrame(trades)
            ret = (capital / 1000 - 1) * 100
            lt = tdf[tdf['dir'] == 'LONG']
            st = tdf[tdf['dir'] == 'SHORT']
            lt_tp = len(lt[lt['exit_type'] == 'TP'])
            st_tp = len(st[st['exit_type'] == 'TP'])
            total_tp = lt_tp + st_tp
            wr = total_tp / len(trades) * 100 if trades else 0

            logger.info(f"  {label}:")
            logger.info(f"    Trades: {len(trades)} (LONG:{len(lt)}, SHORT:{len(st)})")
            if len(lt) > 0:
                logger.info(f"    LONG WR: {lt_tp/len(lt)*100:.0f}% ({lt_tp}/{len(lt)})")
            if len(st) > 0:
                logger.info(f"    SHORT WR: {st_tp/len(st)*100:.0f}% ({st_tp}/{len(st)})")
            logger.info(f"    Total WR: {wr:.1f}%")
            logger.info(f"    Return: {ret:+.2f}%")
            logger.info(f"    Capital: $1000 -> ${capital:.2f}")
            logger.info(f"    Avg PnL: {tdf['pnl'].mean():+.2f}%")
            logger.info("")

            tdf.to_csv(RESULTS_DIR / f'eth_backtest_{label.lower().replace("+", "_")}.csv', index=False)
        else:
            logger.info(f"  {label}: No trades")

    # Winner
    if cap_meta > cap_cnn:
        diff = cap_meta - cap_cnn
        logger.info(f"  >>> CNN+Meta WINS by ${diff:.2f} ({(diff/cap_cnn)*100:+.1f}%)")
    else:
        diff = cap_cnn - cap_meta
        logger.info(f"  >>> CNN-only WINS by ${diff:.2f} ({(diff/cap_meta)*100:+.1f}%)")


if __name__ == "__main__":
    backtest()
