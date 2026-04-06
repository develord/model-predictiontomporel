"""
ADA Meta Threshold Optimizer V3
================================
Grid search on CNN + Meta thresholds to maximize win rate.
Backtest: Q1 2026 (15min intraday sim)

Usage:
    python 07_optimize_meta_thresholds.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import joblib
import json
import logging
from datetime import timedelta

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR / 'scripts'))
from direction_prediction_model import CNNDirectionModel, DeepCNNShortModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = BASE_DIR / 'data' / 'cache'
LONG_MODEL_DIR = BASE_DIR / 'models'
SHORT_MODEL_DIR = BASE_DIR / 'models_short'
META_MODEL_DIR = BASE_DIR / 'models_meta'

POSITION_SIZE = 0.95
TRADING_FEE = 0.001
SLIPPAGE = 0.0005
USE_DYNAMIC_TP_SL = True
TP_PCT = 0.012  # Symmetric fallback (matches training)
SL_PCT = 0.012

BACKTEST_START = '2026-01-01'
BACKTEST_END = '2026-04-03'


def load_cnn(model_dir, model_file):
    path = model_dir / model_file
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    feat_dim = ckpt.get('feature_dim', 99)
    seq_len = ckpt.get('sequence_length', 30)
    is_deep = any('conv3_1' in k or 'conv9_1' in k for k in ckpt['model_state_dict'].keys())
    if is_deep:
        model = DeepCNNShortModel(feature_dim=feat_dim, sequence_length=seq_len, dropout=0.35)
    else:
        model = CNNDirectionModel(feature_dim=feat_dim, sequence_length=seq_len, dropout=0.4)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, seq_len, ckpt


def get_tp_sl(row, entry, direction, s_tp=0.020, s_sl=0.010):
    """ATR-based symmetric TP/SL for both LONG and SHORT (matches training labels)"""
    atr = row.get('1d_atr_14', None)
    ATR_MULT = 1.5  # Must match training
    if USE_DYNAMIC_TP_SL and atr and pd.notna(atr) and atr > 0:
        tp_m = min(max(ATR_MULT * atr / entry, 0.008), 0.04)
        sl_m = tp_m  # Symmetric
    else:
        tp_m, sl_m = TP_PCT, SL_PCT
    if direction == 'LONG':
        return entry * (1 + tp_m), entry * (1 - sl_m)
    else:
        return entry * (1 - tp_m), entry * (1 + sl_m)


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
    feat = {
        'long_conf': long_conf, 'long_dir': long_dir,
        'short_conf': short_conf, 'short_dir': short_dir,
        'long_prob_spread': l_p1 - l_p0, 'short_prob_spread': s_p1 - s_p0,
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


def precompute_predictions(long_model, short_model, long_feat, short_feat,
                           long_seq, short_seq, long_temp, short_temp,
                           df_wide, df_test, t_start):
    """Precompute all CNN + meta predictions once"""
    seq = max(long_seq, short_seq)
    predictions = []

    for i in range(len(df_test)):
        wi = t_start + i
        if wi < seq:
            predictions.append(None)
            continue

        row = df_test.iloc[i]

        lx = torch.tensor(long_feat[wi - long_seq:wi], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            l_logits = long_model(lx)
            l_probs = torch.softmax(l_logits / long_temp, dim=1).squeeze()
        l_conf, l_dir = l_probs.max(0)

        sx = torch.tensor(short_feat[wi - short_seq:wi], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            s_logits = short_model(sx)
            s_probs = torch.softmax(s_logits / short_temp, dim=1).squeeze()
        s_conf, s_dir = s_probs.max(0)

        mf = build_meta_features(
            row, l_conf.item(), l_dir.item(), s_conf.item(), s_dir.item(),
            l_probs[0].item(), l_probs[1].item(), s_probs[0].item(), s_probs[1].item()
        )

        predictions.append({
            'row': row,
            'l_dir': l_dir.item(), 'l_conf': l_conf.item(),
            's_dir': s_dir.item(), 's_conf': s_conf.item(),
            'meta_feat': mf,
        })

    return predictions


def precompute_trade_outcomes(predictions, df_15m, df_test, short_tp, short_sl,
                              meta_long, meta_short, meta_feature_cols):
    """Pre-compute trade outcomes for each day (LONG and SHORT) once."""
    outcomes = []
    for i in range(len(df_test)):
        pred = predictions[i]
        if pred is None:
            outcomes.append(None)
            continue

        row = pred['row']
        date = row['date']
        entry_long = row['close'] * (1 + SLIPPAGE)
        entry_short = row['close'] * (1 - SLIPPAGE)
        tp_l, sl_l = get_tp_sl(row, entry_long, 'LONG')
        tp_s, sl_s = get_tp_sl(row, entry_short, 'SHORT', short_tp, short_sl)

        res_l = sim_trade(entry_long, date, df_15m, tp_l, sl_l, 'LONG')
        pnl_l = (res_l['price'] / entry_long - 1) * 100
        res_s = sim_trade(entry_short, date, df_15m, tp_s, sl_s, 'SHORT')
        pnl_s = (entry_short / res_s['price'] - 1) * 100

        # Meta probabilities
        mf_vec = np.array([[pred['meta_feat'].get(c, 0) for c in meta_feature_cols]])
        meta_long_prob = meta_long.predict_proba(mf_vec)[0][1] if meta_long is not None else 1.0
        meta_short_prob = meta_short.predict_proba(mf_vec)[0][1] if meta_short is not None else 1.0

        outcomes.append({
            'date': date,
            'l_dir': pred['l_dir'], 'l_conf': pred['l_conf'],
            's_dir': pred['s_dir'], 's_conf': pred['s_conf'],
            'meta_long_prob': meta_long_prob, 'meta_short_prob': meta_short_prob,
            'long_exit_type': res_l['type'], 'long_pnl': pnl_l,
            'short_exit_type': res_s['type'], 'short_pnl': pnl_s,
        })
    return outcomes


def run_backtest_fast(outcomes, long_cnn_th, short_cnn_th, long_meta_th, short_meta_th,
                      max_consec=3, cooldown_days=3):
    """Fast backtest using pre-computed outcomes."""
    capital = 1000.0
    trades = []
    long_consec, short_consec = 0, 0
    long_cool, short_cool = None, None

    for o in outcomes:
        if o is None:
            continue
        date = o['date']

        # LONG
        if o['l_dir'] == 1 and o['l_conf'] >= long_cnn_th:
            if long_cool and date <= long_cool:
                pass
            else:
                take = o['meta_long_prob'] >= long_meta_th if long_meta_th > 0 else True
                if take:
                    pos_val = capital * POSITION_SIZE
                    fee = pos_val * TRADING_FEE * 2
                    capital += pos_val * o['long_pnl'] / 100 - fee
                    trades.append({'dir': 'LONG', 'exit_type': o['long_exit_type'],
                                   'pnl': o['long_pnl'], 'date': date})
                    if o['long_exit_type'] == 'SL':
                        long_consec += 1
                        if long_consec >= max_consec:
                            long_cool = date + timedelta(days=cooldown_days)
                    else:
                        long_consec = 0
                    continue

        # SHORT
        if o['s_dir'] == 1 and o['s_conf'] >= short_cnn_th:
            if short_cool and date <= short_cool:
                pass
            else:
                take = o['meta_short_prob'] >= short_meta_th if short_meta_th > 0 else True
                if take:
                    pos_val = capital * POSITION_SIZE
                    fee = pos_val * TRADING_FEE * 2
                    capital += pos_val * o['short_pnl'] / 100 - fee
                    trades.append({'dir': 'SHORT', 'exit_type': o['short_exit_type'],
                                   'pnl': o['short_pnl'], 'date': date})
                    if o['short_exit_type'] == 'SL':
                        short_consec += 1
                        if short_consec >= max_consec:
                            short_cool = date + timedelta(days=cooldown_days)
                    else:
                        short_consec = 0
                    continue

    return capital, trades


def optimize():
    logger.info(f"\n{'='*70}")
    logger.info(f"ADA META THRESHOLD OPTIMIZER V3")
    logger.info(f"{'='*70}\n")

    # Load everything
    long_model, long_seq, long_ckpt = load_cnn(LONG_MODEL_DIR, 'ADA_direction_model.pt')
    short_model, short_seq, short_ckpt = load_cnn(SHORT_MODEL_DIR, 'ADA_short_model.pt')
    long_temp = long_ckpt.get('temperature', 1.0)
    short_temp = short_ckpt.get('temperature', 1.0)

    meta_long = joblib.load(META_MODEL_DIR / 'ADA_meta_long.joblib')
    meta_short = joblib.load(META_MODEL_DIR / 'ADA_meta_short.joblib')
    with open(META_MODEL_DIR / 'meta_features.json') as f:
        meta_feature_cols = json.load(f)

    with open(BASE_DIR / 'required_features.json') as f:
        long_feature_cols = json.load(f)
    with open(SHORT_MODEL_DIR / 'short_features.json') as f:
        short_feature_cols = json.load(f)

    long_scaler = joblib.load(LONG_MODEL_DIR / 'feature_scaler.joblib')
    short_scaler = joblib.load(SHORT_MODEL_DIR / 'feature_scaler_short.joblib')

    short_tp = short_ckpt.get('short_tp_pct', 0.020)
    short_sl = short_ckpt.get('short_sl_pct', 0.010)

    df = pd.read_csv(DATA_DIR / 'ada_features.csv')
    df['date'] = pd.to_datetime(df['date'])

    sys.path.insert(0, str(BASE_DIR))
    try:
        from importlib import import_module
        short_train = import_module('03_train_short_model')
        df = short_train.add_bear_features(df)
    except Exception as e:
        logger.warning(f"Could not add bear features: {e}")

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

    # Load 15min
    cache = DATA_DIR / 'ada_15min_q1_2026.csv'
    df_15m = pd.read_csv(cache, parse_dates=['timestamp'])
    df_15m.set_index('timestamp', inplace=True)

    logger.info(f"Test: {len(df_test)} days | 15min: {len(df_15m)} candles")

    # Precompute predictions
    logger.info("Precomputing CNN predictions...")
    predictions = precompute_predictions(
        long_model, short_model, long_feat, short_feat,
        long_seq, short_seq, long_temp, short_temp,
        df_wide, df_test, t_start
    )
    logger.info("Done. Pre-computing trade outcomes...")

    outcomes = precompute_trade_outcomes(
        predictions, df_15m, df_test, short_tp, short_sl,
        meta_long, meta_short, meta_feature_cols
    )
    logger.info(f"Pre-computed {sum(1 for o in outcomes if o is not None)} day outcomes.")
    logger.info("Starting grid search...\n")

    # Grid search
    long_cnn_range = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    short_cnn_range = [0.50, 0.55, 0.60, 0.65]
    long_meta_range = [0.0, 0.40, 0.45, 0.50, 0.52, 0.55]  # 0.0 = no meta
    short_meta_range = [0.0, 0.40, 0.45, 0.48, 0.50]
    cooldown_range = [2, 3, 5]
    consec_range = [2, 3]

    results = []
    total = (len(long_cnn_range) * len(short_cnn_range) * len(long_meta_range) *
             len(short_meta_range) * len(cooldown_range) * len(consec_range))
    logger.info(f"Testing {total} combinations...")

    count = 0
    for lc in long_cnn_range:
        for sc in short_cnn_range:
            for lm in long_meta_range:
                for sm in short_meta_range:
                    for cd in cooldown_range:
                        for mc in consec_range:
                            count += 1

                            capital, trades = run_backtest_fast(
                                outcomes, lc, sc, lm, sm, mc, cd
                            )

                            if len(trades) < 3:
                                continue

                            tdf = pd.DataFrame(trades)
                            tp_sl_trades = tdf[tdf['exit_type'].isin(['TP', 'SL'])]
                            if len(tp_sl_trades) < 3:
                                continue

                            n_tp = len(tp_sl_trades[tp_sl_trades['exit_type'] == 'TP'])
                            wr = n_tp / len(tp_sl_trades) * 100
                            ret = (capital / 1000 - 1) * 100
                            avg_pnl = tdf['pnl'].mean()

                            lt = tdf[tdf['dir'] == 'LONG']
                            st = tdf[tdf['dir'] == 'SHORT']
                            lt_tpsl = lt[lt['exit_type'].isin(['TP', 'SL'])]
                            st_tpsl = st[st['exit_type'].isin(['TP', 'SL'])]
                            lt_wr = (lt_tpsl['exit_type'] == 'TP').mean() * 100 if len(lt_tpsl) > 0 else 0
                            st_wr = (st_tpsl['exit_type'] == 'TP').mean() * 100 if len(st_tpsl) > 0 else 0

                            results.append({
                                'long_cnn': lc, 'short_cnn': sc,
                                'long_meta': lm, 'short_meta': sm,
                                'cooldown': cd, 'max_consec': mc,
                                'trades': len(trades), 'tp_sl_trades': len(tp_sl_trades),
                                'wr': wr, 'return': ret, 'avg_pnl': avg_pnl,
                                'long_trades': len(lt), 'short_trades': len(st),
                                'long_wr': lt_wr, 'short_wr': st_wr,
                                'capital': capital,
                            })

    rdf = pd.DataFrame(results)
    logger.info(f"\nTested {count} combinations, {len(rdf)} with 3+ trades\n")

    # === RESULTS ===
    logger.info(f"{'='*70}")
    logger.info(f"TOP CONFIGS BY WIN RATE (min 5 TP/SL trades)")
    logger.info(f"{'='*70}\n")

    rdf_valid = rdf[rdf['tp_sl_trades'] >= 5].copy()
    rdf_valid = rdf_valid.sort_values(['wr', 'return'], ascending=[False, False])

    for idx, row in rdf_valid.head(20).iterrows():
        meta_str = ""
        if row['long_meta'] > 0:
            meta_str += f"LMeta>={row['long_meta']:.0%} "
        if row['short_meta'] > 0:
            meta_str += f"SMeta>={row['short_meta']:.0%} "
        if not meta_str:
            meta_str = "NoMeta "

        logger.info(
            f"  WR:{row['wr']:5.1f}% | Ret:{row['return']:+6.1f}% | "
            f"Trades:{row['trades']:2.0f} (TP/SL:{row['tp_sl_trades']:2.0f}) | "
            f"L:{row['long_trades']:2.0f}({row['long_wr']:.0f}%) S:{row['short_trades']:2.0f}({row['short_wr']:.0f}%) | "
            f"LCNN>={row['long_cnn']:.0%} SCNN>={row['short_cnn']:.0%} {meta_str}| "
            f"CD:{row['cooldown']:.0f}d MC:{row['max_consec']:.0f}"
        )

    logger.info(f"\n{'='*70}")
    logger.info(f"TOP CONFIGS BY RETURN (min 5 TP/SL trades, WR >= 50%)")
    logger.info(f"{'='*70}\n")

    rdf_wr50 = rdf_valid[rdf_valid['wr'] >= 50].sort_values('return', ascending=False)
    for idx, row in rdf_wr50.head(20).iterrows():
        meta_str = ""
        if row['long_meta'] > 0:
            meta_str += f"LMeta>={row['long_meta']:.0%} "
        if row['short_meta'] > 0:
            meta_str += f"SMeta>={row['short_meta']:.0%} "
        if not meta_str:
            meta_str = "NoMeta "

        logger.info(
            f"  WR:{row['wr']:5.1f}% | Ret:{row['return']:+6.1f}% | "
            f"Trades:{row['trades']:2.0f} (TP/SL:{row['tp_sl_trades']:2.0f}) | "
            f"L:{row['long_trades']:2.0f}({row['long_wr']:.0f}%) S:{row['short_trades']:2.0f}({row['short_wr']:.0f}%) | "
            f"LCNN>={row['long_cnn']:.0%} SCNN>={row['short_cnn']:.0%} {meta_str}| "
            f"CD:{row['cooldown']:.0f}d MC:{row['max_consec']:.0f}"
        )

    logger.info(f"\n{'='*70}")
    logger.info(f"BEST BALANCED (WR >= 55% AND Return > 0%)")
    logger.info(f"{'='*70}\n")

    rdf_balanced = rdf_valid[(rdf_valid['wr'] >= 55) & (rdf_valid['return'] > 0)]
    rdf_balanced = rdf_balanced.sort_values('return', ascending=False)
    for idx, row in rdf_balanced.head(10).iterrows():
        meta_str = ""
        if row['long_meta'] > 0:
            meta_str += f"LMeta>={row['long_meta']:.0%} "
        if row['short_meta'] > 0:
            meta_str += f"SMeta>={row['short_meta']:.0%} "
        if not meta_str:
            meta_str = "NoMeta "

        logger.info(
            f"  WR:{row['wr']:5.1f}% | Ret:{row['return']:+6.1f}% | "
            f"Trades:{row['trades']:2.0f} (TP/SL:{row['tp_sl_trades']:2.0f}) | "
            f"L:{row['long_trades']:2.0f}({row['long_wr']:.0f}%) S:{row['short_trades']:2.0f}({row['short_wr']:.0f}%) | "
            f"LCNN>={row['long_cnn']:.0%} SCNN>={row['short_cnn']:.0%} {meta_str}| "
            f"CD:{row['cooldown']:.0f}d MC:{row['max_consec']:.0f}"
        )

    # Save
    RESULTS_DIR = BASE_DIR / 'results'
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rdf.to_csv(RESULTS_DIR / 'threshold_optimization.csv', index=False)
    logger.info(f"\nAll results saved to results/threshold_optimization.csv")


if __name__ == "__main__":
    optimize()
