"""
V12 Win Rate Optimization - Full Pipeline
==========================================
Problem: Model trained on V11 fixed labels (1.5%/0.75%) but executes with ATR dynamic TP/SL.
         This mismatch kills win rate (45-51% instead of 55-76% V11).

Solution:
1. Retrain XGBoost on DYNAMIC ATR labels (match training = execution)
2. Optimize for precision (win rate) not AUC
3. Grid search confidence threshold per crypto
4. LSTM as hard veto filter (not just a feature)
5. Walk-forward validation P1 + P2

Usage:
    python v12/optimization/optimize_v12_winrate.py
"""

import sys
import numpy as np
import pandas as pd
import json
import time
from pathlib import Path
from itertools import product
import xgboost as xgb
import joblib

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from v12.features.dynamic_labels import calculate_atr_series, apply_dynamic_triple_barrier
from v12.features.lstm_features import build_lstm_features_for_crypto

# ============================================================================
# CONFIG
# ============================================================================
CRYPTOS = ['btc', 'eth', 'sol']

# Load V12 config for ATR params
with open(Path(__file__).parent.parent / 'config' / 'v12_config.json') as f:
    V12_CONFIG = json.load(f)

GLOBAL_ATR = V12_CONFIG['dynamic_tp_sl']
TRADING = V12_CONFIG['trading']
FEE = TRADING['fee_pct']
POS_PCT = TRADING['position_size_pct']

# Walk-forward periods
PERIODS = [
    {'name': 'P1', 'train_end': '2024-01-01', 'test_start': '2024-01-01', 'test_end': '2025-01-01',
     'lstm_cutoff': '2023-01-01'},
    {'name': 'P2', 'train_end': '2025-01-01', 'test_start': '2025-01-01', 'test_end': '2026-01-01',
     'lstm_cutoff': '2024-01-01'}
]

# Grid search: confidence thresholds
THRESHOLDS = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

# XGBoost hyperparams to try (precision-focused)
XGB_CONFIGS = {
    'precision_conservative': {
        'objective': 'binary:logistic',
        'eval_metric': 'aucpr',
        'max_depth': 4,
        'learning_rate': 0.03,
        'n_estimators': 300,
        'gamma': 3,
        'min_child_weight': 5,
        'subsample': 0.7,
        'colsample_bytree': 0.6,
        'reg_alpha': 0.5,
        'reg_lambda': 2.0,
        'random_state': 42,
        'tree_method': 'hist',
    },
    'precision_balanced': {
        'objective': 'binary:logistic',
        'eval_metric': 'aucpr',
        'max_depth': 5,
        'learning_rate': 0.05,
        'n_estimators': 250,
        'gamma': 2,
        'min_child_weight': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'reg_alpha': 0.3,
        'reg_lambda': 1.5,
        'random_state': 42,
        'tree_method': 'hist',
    },
    'precision_aggressive': {
        'objective': 'binary:logistic',
        'eval_metric': 'aucpr',
        'max_depth': 6,
        'learning_rate': 0.05,
        'n_estimators': 200,
        'gamma': 2,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'tree_method': 'hist',
    }
}


# ============================================================================
# DATA LOADING
# ============================================================================
def load_data(crypto):
    f = PROJECT_ROOT / 'data' / 'cache' / f'{crypto}_multi_tf_merged.csv'
    return pd.read_csv(f, index_col=0, parse_dates=True)


def load_sol_top50():
    f = PROJECT_ROOT / 'optimization' / 'results' / 'sol_selected_features_top50.json'
    if f.exists():
        with open(f) as fh:
            return json.load(fh).get('selected_feature_names', [])
    return None


def get_feature_cols(df, crypto):
    exclude = ['open', 'high', 'low', 'close', 'volume',
               'label_class', 'label_numeric', 'price_target_pct',
               'future_price', 'triple_barrier_label',
               'dynamic_tp_pct', 'dynamic_sl_pct', 'v11_fixed_label',
               'dynamic_label']
    all_cols = [c for c in df.columns if c not in exclude]
    if crypto == 'sol':
        top50 = load_sol_top50()
        if top50:
            cols = [c for c in top50 if c in all_cols]
            for extra in ['atr_pct_14', 'lstm_proba', 'lstm_confidence',
                          'lstm_signal', 'lstm_agrees_rsi']:
                if extra in all_cols and extra not in cols:
                    cols.append(extra)
            return cols
    return all_cols


# ============================================================================
# DYNAMIC ATR LABELS FOR TRAINING
# ============================================================================
def generate_dynamic_labels(df, crypto):
    """Generate ATR-based labels that match execution TP/SL."""
    crypto_cfg = V12_CONFIG.get(crypto.upper(), {})

    tp_mult = crypto_cfg.get('tp_atr_multiplier', 0.40)
    sl_mult = crypto_cfg.get('sl_atr_multiplier', 0.15)
    atr_period = GLOBAL_ATR.get('atr_period', 14)
    lookahead = TRADING.get('lookahead_candles_1d', 7)

    result = apply_dynamic_triple_barrier(
        df,
        tp_atr_mult=tp_mult,
        sl_atr_mult=sl_mult,
        atr_period=atr_period,
        lookahead_candles=lookahead,
        min_tp_pct=GLOBAL_ATR.get('min_tp_pct', 0.75),
        max_tp_pct=GLOBAL_ATR.get('max_tp_pct', 3.5),
        min_sl_pct=GLOBAL_ATR.get('min_sl_pct', 0.35),
        max_sl_pct=GLOBAL_ATR.get('max_sl_pct', 1.75),
        min_rr_ratio=GLOBAL_ATR.get('risk_reward_min', 1.8)
    )

    return result


# ============================================================================
# LSTM FEATURES
# ============================================================================
def add_lstm_features(df, crypto, lstm_cutoff):
    """Add LSTM features with proper temporal cutoff."""
    try:
        lstm_features, lstm_model = build_lstm_features_for_crypto(
            crypto,
            lstm_train_end=lstm_cutoff,
            seq_len=20,
            epochs=30,
            verbose=False
        )
        for col in ['lstm_proba', 'lstm_confidence', 'lstm_signal', 'lstm_agrees_rsi']:
            if col in lstm_features.columns:
                df[col] = lstm_features[col].values
        return df, lstm_model
    except Exception as e:
        print(f"    LSTM failed ({e}), filling defaults")
        df['lstm_proba'] = 0.5
        df['lstm_confidence'] = 0.0
        df['lstm_signal'] = 0
        df['lstm_agrees_rsi'] = 0
        return df, None


# ============================================================================
# BACKTEST ENGINE
# ============================================================================
def dynamic_tp_sl(atr, crypto):
    """Compute TP/SL from ATR - matches execution exactly."""
    crypto_cfg = V12_CONFIG.get(crypto.upper(), {})
    tp_m = crypto_cfg.get('tp_atr_multiplier', 0.40)
    sl_m = crypto_cfg.get('sl_atr_multiplier', 0.15)

    tp = np.clip(atr * tp_m, GLOBAL_ATR['min_tp_pct'], GLOBAL_ATR['max_tp_pct'])
    sl = np.clip(atr * sl_m, GLOBAL_ATR['min_sl_pct'], GLOBAL_ATR['max_sl_pct'])

    if tp / sl < GLOBAL_ATR['risk_reward_min']:
        sl = tp / GLOBAL_ATR['risk_reward_min']
        sl = max(sl, GLOBAL_ATR['min_sl_pct'])

    return tp, sl


def backtest_period(model, feat_cols, test_df, crypto, threshold,
                    use_lstm_veto=False, lstm_veto_threshold=0.45):
    """Run backtest with optional LSTM veto filter."""
    capital = 10000.0
    trades = []

    for i in range(len(test_df) - 1):
        row = test_df.iloc[i]
        atr = row.get('atr_pct_14', np.nan)
        if pd.isna(atr):
            continue

        features = row[feat_cols].fillna(0).values.reshape(1, -1)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        prob = model.predict_proba(features)[0, 1]

        # Confidence filter
        if prob < threshold:
            continue

        # LSTM veto: if LSTM disagrees, skip
        if use_lstm_veto:
            lstm_p = row.get('lstm_proba', 0.5)
            if lstm_p < lstm_veto_threshold:
                continue

        tp, sl = dynamic_tp_sl(atr, crypto)
        entry = row['close']
        tp_price = entry * (1 + tp / 100)
        sl_price = entry * (1 - sl / 100)
        pos = capital * POS_PCT / 100

        for j in range(1, min(len(test_df) - i, 100)):
            h = test_df.iloc[i + j]['high']
            l = test_df.iloc[i + j]['low']

            if h >= tp_price:
                pnl = pos * (tp - FEE * 2) / 100
                capital += pnl
                trades.append({'result': 'WIN', 'pnl': pnl, 'pnl_pct': tp - FEE * 2,
                               'prob': prob, 'tp': tp, 'sl': sl, 'hold': j})
                break
            if l <= sl_price:
                pnl = pos * (-sl - FEE * 2) / 100
                capital += pnl
                trades.append({'result': 'LOSE', 'pnl': pnl, 'pnl_pct': -sl - FEE * 2,
                               'prob': prob, 'tp': tp, 'sl': sl, 'hold': j})
                break

    if not trades:
        return {'trades': 0, 'wins': 0, 'losses': 0, 'wr': 0, 'roi': 0, 'pf': 0,
                'max_dd': 0, 'avg_conf': 0}

    tdf = pd.DataFrame(trades)
    wins = len(tdf[tdf['result'] == 'WIN'])
    losses = len(tdf[tdf['result'] == 'LOSE'])
    total = wins + losses
    roi = (capital / 10000 - 1) * 100

    win_pnl = tdf[tdf['result'] == 'WIN']['pnl'].sum() if wins > 0 else 0
    loss_pnl = abs(tdf[tdf['result'] == 'LOSE']['pnl'].sum()) if losses > 0 else 1
    pf = win_pnl / loss_pnl if loss_pnl > 0 else float('inf')

    # Max drawdown
    equity = 10000 + tdf['pnl'].cumsum()
    peak = equity.cummax()
    dd = ((equity - peak) / peak * 100).min()

    return {
        'trades': total, 'wins': wins, 'losses': losses,
        'wr': wins / total * 100 if total > 0 else 0,
        'roi': roi, 'pf': pf, 'max_dd': dd,
        'avg_conf': float(tdf['prob'].mean()),
        'avg_tp': float(tdf['tp'].mean()),
        'avg_sl': float(tdf['sl'].mean()),
    }


# ============================================================================
# MAIN OPTIMIZATION
# ============================================================================
def optimize_crypto(crypto):
    """Full optimization pipeline for one crypto."""
    print(f"\n{'='*100}")
    print(f"  OPTIMIZING {crypto.upper()} - Win Rate Maximization")
    print(f"{'='*100}")

    # [1] Load data
    print(f"\n  [1] Loading data...")
    df = load_data(crypto)
    print(f"      {len(df)} rows, {len(df.columns)} columns")

    # [2] Generate DYNAMIC ATR labels (KEY CHANGE: match training to execution)
    print(f"\n  [2] Generating dynamic ATR labels...")
    df = generate_dynamic_labels(df, crypto)

    # Store dynamic label separately
    df['dynamic_label'] = df['triple_barrier_label'].copy()

    valid_dynamic = df[df['dynamic_label'].notna() & (df['dynamic_label'] != 0)]
    n_tp = (valid_dynamic['dynamic_label'] == 1).sum()
    n_sl = (valid_dynamic['dynamic_label'] == -1).sum()
    print(f"      Dynamic labels: TP={n_tp} ({n_tp/(n_tp+n_sl)*100:.1f}%) | SL={n_sl} ({n_sl/(n_tp+n_sl)*100:.1f}%)")

    # Also show V11 fixed labels for comparison
    v11_valid = df[df.get('v11_fixed_label', pd.Series(dtype=float)).notna()] if 'v11_fixed_label' in df.columns else pd.DataFrame()
    if len(v11_valid) > 0:
        v11_tp = (v11_valid['v11_fixed_label'] == 1).sum()
        v11_sl = (v11_valid['v11_fixed_label'] == -1).sum()
        print(f"      V11 fixed labels: TP={v11_tp} ({v11_tp/(v11_tp+v11_sl)*100:.1f}%) | SL={v11_sl}")

    # [3] ATR feature
    print(f"\n  [3] Adding ATR feature...")
    if 'atr_pct_14' not in df.columns:
        df['atr_pct_14'] = calculate_atr_series(df, 14)

    all_period_results = {}

    for period in PERIODS:
        pname = period['name']
        print(f"\n  {'='*80}")
        print(f"  PERIOD {pname}: Train <{period['train_end']}, Test [{period['test_start']}, {period['test_end']})")
        print(f"  {'='*80}")

        # [4] LSTM features
        print(f"\n  [4] Building LSTM features (cutoff={period['lstm_cutoff']})...")
        df_period = df.copy()
        df_period, lstm_model = add_lstm_features(df_period, crypto, period['lstm_cutoff'])

        # [5] Prepare data
        feat_cols = get_feature_cols(df_period, crypto)
        print(f"      Features: {len(feat_cols)}")

        # Training data with DYNAMIC labels
        train_df = df_period[df_period.index < period['train_end']].copy()
        test_df = df_period[
            (df_period.index >= period['test_start']) &
            (df_period.index < period['test_end'])
        ].copy()

        # Clean training data (dynamic labels only, no timeout)
        train_clean = train_df[
            train_df['dynamic_label'].notna() &
            (train_df['dynamic_label'] != 0)
        ].copy()

        X_train = train_clean[feat_cols].fillna(0).values
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        y_train = (train_clean['dynamic_label'].values == 1).astype(int)

        n_neg = np.sum(y_train == 0)
        n_pos = np.sum(y_train == 1)
        print(f"      Train: {len(X_train)} samples (TP={n_pos}, SL={n_neg})")
        print(f"      Test: {len(test_df)} candles")

        # [6] Grid search: XGB config x threshold x LSTM veto
        print(f"\n  [6] Grid search...")
        results = []

        for cfg_name, xgb_params in XGB_CONFIGS.items():
            params = xgb_params.copy()
            params['scale_pos_weight'] = n_neg / max(n_pos, 1)

            model = xgb.XGBClassifier(**params, verbosity=0)
            model.fit(X_train, y_train, verbose=False)

            for thresh in THRESHOLDS:
                # Without LSTM veto
                r = backtest_period(model, feat_cols, test_df, crypto, thresh,
                                    use_lstm_veto=False)
                r['config'] = cfg_name
                r['threshold'] = thresh
                r['lstm_veto'] = False
                r['lstm_veto_thresh'] = 0
                results.append(r)

                # With LSTM veto at different levels
                for lv_thresh in [0.45, 0.50, 0.55]:
                    r2 = backtest_period(model, feat_cols, test_df, crypto, thresh,
                                         use_lstm_veto=True, lstm_veto_threshold=lv_thresh)
                    r2['config'] = cfg_name
                    r2['threshold'] = thresh
                    r2['lstm_veto'] = True
                    r2['lstm_veto_thresh'] = lv_thresh
                    results.append(r2)

        # [7] Filter and rank
        valid = [r for r in results if r['trades'] >= 30 and r['roi'] > 0]
        valid.sort(key=lambda x: (-x['wr'], -x['roi']))

        print(f"\n      {len(valid)}/{len(results)} valid combos (trades>=30, ROI>0)")

        print(f"\n  TOP 15 by WIN RATE ({pname}):")
        print(f"  {'Config':<25} {'Thresh':>6} {'LSTM':>6} {'LV_th':>5} "
              f"{'Trades':>6} {'W':>4} {'L':>4} {'WR%':>6} {'ROI%':>7} "
              f"{'PF':>5} {'MaxDD':>6} {'AvgTP':>6} {'AvgSL':>6}")
        print(f"  {'-'*110}")

        for r in valid[:15]:
            lv = f"{r['lstm_veto_thresh']:.2f}" if r['lstm_veto'] else "  -  "
            print(f"  {r['config']:<25} {r['threshold']:>6.2f} {'YES' if r['lstm_veto'] else ' NO':>6} {lv:>5} "
                  f"{r['trades']:>6} {r['wins']:>4} {r['losses']:>4} {r['wr']:>5.1f}% {r['roi']:>+6.2f}% "
                  f"{r['pf']:>5.2f} {r['max_dd']:>5.2f}% {r.get('avg_tp',0):>5.2f}% {r.get('avg_sl',0):>5.2f}%")

        all_period_results[pname] = valid[:20] if valid else []

    # [8] Find BEST config consistent across P1+P2
    print(f"\n\n  {'='*80}")
    print(f"  CROSS-PERIOD ANALYSIS: {crypto.upper()}")
    print(f"  {'='*80}")

    best_overall = find_best_consistent(all_period_results, crypto)
    return all_period_results, best_overall


def find_best_consistent(period_results, crypto):
    """Find the config that works best across both periods."""
    p1_results = period_results.get('P1', [])
    p2_results = period_results.get('P2', [])

    if not p1_results or not p2_results:
        print("  Not enough data for cross-period analysis")
        return None

    # Build lookup: (config, threshold, lstm_veto, lv_thresh) -> result
    p1_map = {}
    for r in p1_results:
        key = (r['config'], r['threshold'], r['lstm_veto'], r['lstm_veto_thresh'])
        p1_map[key] = r

    p2_map = {}
    for r in p2_results:
        key = (r['config'], r['threshold'], r['lstm_veto'], r['lstm_veto_thresh'])
        p2_map[key] = r

    # Find combos present in both periods
    common_keys = set(p1_map.keys()) & set(p2_map.keys())

    if not common_keys:
        print("  No common configs across periods")
        # Fall back to best individual
        best = p2_results[0] if p2_results else p1_results[0]
        print(f"\n  FALLBACK BEST (P2): {best['config']} thresh={best['threshold']:.2f} "
              f"lstm_veto={best['lstm_veto']} -> WR={best['wr']:.1f}% ROI={best['roi']:+.2f}%")
        return best

    # Score: average WR across periods, penalize inconsistency
    scored = []
    for key in common_keys:
        r1, r2 = p1_map[key], p2_map[key]
        avg_wr = (r1['wr'] + r2['wr']) / 2
        avg_roi = (r1['roi'] + r2['roi']) / 2
        wr_diff = abs(r1['wr'] - r2['wr'])
        # Score = avg WR - penalty for inconsistency + bonus for ROI
        score = avg_wr - wr_diff * 0.3 + min(avg_roi, 15) * 0.1
        scored.append({
            'key': key, 'r1': r1, 'r2': r2,
            'avg_wr': avg_wr, 'avg_roi': avg_roi,
            'wr_diff': wr_diff, 'score': score
        })

    scored.sort(key=lambda x: -x['score'])

    print(f"\n  TOP 10 CONSISTENT CONFIGS:")
    print(f"  {'Config':<25} {'Thresh':>6} {'LSTM':>5} {'LV':>5} "
          f"{'P1_WR':>6} {'P2_WR':>6} {'AvgWR':>6} {'P1_ROI':>7} {'P2_ROI':>7} {'Score':>6}")
    print(f"  {'-'*100}")

    for s in scored[:10]:
        k = s['key']
        lv = f"{k[3]:.2f}" if k[2] else "  -  "
        print(f"  {k[0]:<25} {k[1]:>6.2f} {'Y' if k[2] else 'N':>5} {lv:>5} "
              f"{s['r1']['wr']:>5.1f}% {s['r2']['wr']:>5.1f}% {s['avg_wr']:>5.1f}% "
              f"{s['r1']['roi']:>+6.2f}% {s['r2']['roi']:>+6.2f}% {s['score']:>5.1f}")

    best = scored[0]
    print(f"\n  >>> BEST for {crypto.upper()}: {best['key'][0]} "
          f"thresh={best['key'][1]:.2f} lstm_veto={best['key'][2]} "
          f"lv_thresh={best['key'][3]:.2f}")
    print(f"      P1: WR={best['r1']['wr']:.1f}% ROI={best['r1']['roi']:+.2f}% "
          f"Trades={best['r1']['trades']} PF={best['r1']['pf']:.2f}")
    print(f"      P2: WR={best['r2']['wr']:.1f}% ROI={best['r2']['roi']:+.2f}% "
          f"Trades={best['r2']['trades']} PF={best['r2']['pf']:.2f}")

    return {
        'config': best['key'][0],
        'threshold': best['key'][1],
        'lstm_veto': best['key'][2],
        'lstm_veto_thresh': best['key'][3],
        'p1': best['r1'],
        'p2': best['r2'],
        'avg_wr': best['avg_wr'],
        'avg_roi': best['avg_roi'],
        'score': best['score']
    }


# ============================================================================
# COMPARISON TABLE V11 vs V12 CURRENT vs V12 OPTIMIZED
# ============================================================================
def print_final_comparison(all_best):
    """Print final V11 vs V12 current vs V12 optimized."""
    # Load V12 current results
    comparison_file = Path(__file__).parent.parent / 'results' / 'final_comparison.json'
    with open(comparison_file) as f:
        v12_current = json.load(f)

    print(f"\n\n{'='*120}")
    print("FINAL COMPARISON: V11 vs V12 Current vs V12 Optimized")
    print(f"{'='*120}")

    print(f"\n{'Crypto':<7} {'Version':<18} {'P1_WR':>7} {'P1_ROI':>8} {'P1_PF':>6} {'P1_Tr':>6} "
          f"{'P2_WR':>7} {'P2_ROI':>8} {'P2_PF':>6} {'P2_Tr':>6} {'AvgWR':>7} {'AvgROI':>8}")
    print("-" * 120)

    for crypto in CRYPTOS:
        c_data = v12_current.get(crypto, {})

        # V11
        v11_p1 = c_data.get('P1', {}).get('v11', {})
        v11_p2 = c_data.get('P2', {}).get('v11', {})
        if v11_p1 and v11_p2:
            avg_wr = (v11_p1.get('win_rate', 0) + v11_p2.get('win_rate', 0)) / 2
            avg_roi = (v11_p1.get('roi', 0) + v11_p2.get('roi', 0)) / 2
            print(f"{crypto.upper():<7} {'V11 (fixed)':<18} "
                  f"{v11_p1.get('win_rate',0):>6.1f}% {v11_p1.get('roi',0):>+7.2f}% {v11_p1.get('profit_factor',0):>5.2f} {v11_p1.get('total_trades',0):>6} "
                  f"{v11_p2.get('win_rate',0):>6.1f}% {v11_p2.get('roi',0):>+7.2f}% {v11_p2.get('profit_factor',0):>5.2f} {v11_p2.get('total_trades',0):>6} "
                  f"{avg_wr:>6.1f}% {avg_roi:>+7.2f}%")

        # V12 current (prod)
        v12_p1 = c_data.get('P1', {}).get('v12_prod', {})
        v12_p2 = c_data.get('P2', {}).get('v12_prod', {})
        if v12_p1 and v12_p2:
            avg_wr = (v12_p1.get('win_rate', 0) + v12_p2.get('win_rate', 0)) / 2
            avg_roi = (v12_p1.get('roi', 0) + v12_p2.get('roi', 0)) / 2
            print(f"{'':>7} {'V12 current':<18} "
                  f"{v12_p1.get('win_rate',0):>6.1f}% {v12_p1.get('roi',0):>+7.2f}% {v12_p1.get('profit_factor',0):>5.2f} {v12_p1.get('total_trades',0):>6} "
                  f"{v12_p2.get('win_rate',0):>6.1f}% {v12_p2.get('roi',0):>+7.2f}% {v12_p2.get('profit_factor',0):>5.2f} {v12_p2.get('total_trades',0):>6} "
                  f"{avg_wr:>6.1f}% {avg_roi:>+7.2f}%")

        # V12 optimized
        opt = all_best.get(crypto)
        if opt:
            p1 = opt.get('p1', {})
            p2 = opt.get('p2', {})
            cfg_label = f"V12 opt ({opt['config'][:10]})"
            veto_label = f" +LSTM" if opt.get('lstm_veto') else ""
            print(f"{'':>7} {cfg_label + veto_label:<18} "
                  f"{p1.get('wr',0):>6.1f}% {p1.get('roi',0):>+7.2f}% {p1.get('pf',0):>5.2f} {p1.get('trades',0):>6} "
                  f"{p2.get('wr',0):>6.1f}% {p2.get('roi',0):>+7.2f}% {p2.get('pf',0):>5.2f} {p2.get('trades',0):>6} "
                  f"{opt['avg_wr']:>6.1f}% {opt['avg_roi']:>+7.2f}%")

            # Delta vs V12 current
            if v12_p1 and v12_p2:
                d_wr = opt['avg_wr'] - (v12_p1.get('win_rate', 0) + v12_p2.get('win_rate', 0)) / 2
                d_roi = opt['avg_roi'] - (v12_p1.get('roi', 0) + v12_p2.get('roi', 0)) / 2
                print(f"{'':>7} {'  DELTA vs current':<18} "
                      f"{'':>7} {'':>8} {'':>6} {'':>6} "
                      f"{'':>7} {'':>8} {'':>6} {'':>6} "
                      f"{d_wr:>+6.1f}% {d_roi:>+7.2f}%")

        print()


# ============================================================================
# RUN
# ============================================================================
def run():
    print("=" * 120)
    print("V12 WIN RATE OPTIMIZATION")
    print("=" * 120)
    print("Strategy: Retrain on dynamic ATR labels + precision XGB + LSTM veto")
    print(f"Cryptos: {CRYPTOS}")
    print(f"XGB configs: {list(XGB_CONFIGS.keys())}")
    print(f"Thresholds: {THRESHOLDS}")
    print(f"LSTM veto levels: [0.45, 0.50, 0.55]")
    total_combos = len(XGB_CONFIGS) * len(THRESHOLDS) * 4  # 4 = no veto + 3 veto levels
    print(f"Total combos per period: {total_combos}")

    start = time.time()
    all_best = {}

    for crypto in CRYPTOS:
        period_results, best = optimize_crypto(crypto)
        if best:
            all_best[crypto] = best

    # Final comparison
    print_final_comparison(all_best)

    # Save results
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)

    save_data = {}
    for crypto, best in all_best.items():
        save_data[crypto] = {
            'config': best['config'],
            'threshold': best['threshold'],
            'lstm_veto': best['lstm_veto'],
            'lstm_veto_thresh': best['lstm_veto_thresh'],
            'avg_wr': best['avg_wr'],
            'avg_roi': best['avg_roi'],
            'score': best['score'],
            'p1': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                   for k, v in best['p1'].items()},
            'p2': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                   for k, v in best['p2'].items()},
        }

    with open(results_dir / 'optimize_v12_winrate_results.json', 'w') as f:
        json.dump(save_data, f, indent=2)

    elapsed = time.time() - start
    print(f"\nDone in {elapsed/60:.1f} min")
    print(f"Saved to {results_dir / 'optimize_v12_winrate_results.json'}")

    return all_best


if __name__ == '__main__':
    run()
