"""
V12 Walk-Forward Validation - LSTM + ATR Integration
=====================================================
For each period:
1. Train LSTM on data < (test_start - 1 year) [anti-leakage gap]
2. Generate LSTM features for all data
3. Train XGBoost with ATR + LSTM features
4. Backtest with dynamic ATR TP/SL

Periods:
- P1: LSTM train <2023, XGB train <2024, Test 2024
- P2: LSTM train <2024, XGB train <2025, Test 2025

Compare: V11 fixed | V12 ATR-only | V12 ATR+LSTM
"""

import sys
import numpy as np
import pandas as pd
import json
from pathlib import Path
from itertools import product
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from v12.features.dynamic_labels import calculate_atr_series
from v12.features.lstm_features import (
    build_lstm_features_for_crypto, prepare_lstm_features,
    train_lstm, generate_lstm_features, CryptoLSTM, LSTM_FEATURES
)

CRYPTOS = ['btc', 'eth', 'sol']
LSTM_COLS = ['lstm_proba', 'lstm_confidence', 'lstm_signal', 'lstm_agrees_rsi']

PERIODS = [
    {'name': 'P1', 'lstm_end': '2023-01-01', 'xgb_end': '2024-01-01',
     'test_start': '2024-01-01', 'test_end': '2025-01-01',
     'desc': 'LSTM<2023, XGB<2024, Test 2024'},
    {'name': 'P2', 'lstm_end': '2024-01-01', 'xgb_end': '2025-01-01',
     'test_start': '2025-01-01', 'test_end': '2026-01-01',
     'desc': 'LSTM<2024, XGB<2025, Test 2025'}
]

# Optimal ATR config from grid search
ATR_CONFIG = {
    'btc': {'tp_mult': 0.40, 'sl_mult': 0.15},
    'eth': {'tp_mult': 0.40, 'sl_mult': 0.15},
    'sol': {'tp_mult': 0.60, 'sl_mult': 0.15}
}

FEE = 0.1
POS_PCT = 10.0
THRESHOLD = 0.35
MIN_TP, MAX_TP = 0.75, 3.5
MIN_SL, MAX_SL = 0.35, 1.75
MIN_RR = 1.8


def load_data(crypto):
    f = PROJECT_ROOT / 'data' / 'cache' / f'{crypto}_multi_tf_merged.csv'
    return pd.read_csv(f, index_col=0, parse_dates=True)


def load_sol_top50():
    f = PROJECT_ROOT / 'optimization' / 'results' / 'sol_selected_features_top50.json'
    if f.exists():
        with open(f) as fh:
            return json.load(fh).get('selected_feature_names', [])
    return None


def get_features(df, crypto, include_lstm=True):
    exclude = ['open', 'high', 'low', 'close', 'volume',
               'label_class', 'label_numeric', 'price_target_pct',
               'future_price', 'triple_barrier_label']
    all_cols = [c for c in df.columns if c not in exclude]

    if crypto == 'sol':
        top50 = load_sol_top50()
        if top50:
            cols = [c for c in top50 if c in all_cols]
            for extra in ['atr_pct_14'] + (LSTM_COLS if include_lstm else []):
                if extra in all_cols and extra not in cols:
                    cols.append(extra)
            return cols
    return all_cols


def dynamic_tp_sl(atr, tp_m, sl_m):
    tp = np.clip(atr * tp_m, MIN_TP, MAX_TP)
    sl = np.clip(atr * sl_m, MIN_SL, MAX_SL)
    if tp / sl < MIN_RR:
        sl = max(tp / MIN_RR, MIN_SL)
    return tp, sl


def backtest(model, test_df, feat_cols, tp_m, sl_m):
    capital = 10000.0
    trades = []

    for i in range(len(test_df) - 1):
        row = test_df.iloc[i]
        atr = row.get('atr_pct_14', np.nan)
        if pd.isna(atr):
            continue

        features = row[feat_cols].fillna(0).values.reshape(1, -1)
        features = np.nan_to_num(features)
        prob = model.predict_proba(features)[0, 1]

        if prob < THRESHOLD:
            continue

        tp, sl = dynamic_tp_sl(atr, tp_m, sl_m)
        entry = row['close']
        tp_price = entry * (1 + tp / 100)
        sl_price = entry * (1 - sl / 100)
        pos = capital * POS_PCT / 100

        for j in range(1, min(len(test_df) - i, 100)):
            h, l = test_df.iloc[i + j]['high'], test_df.iloc[i + j]['low']
            if h >= tp_price:
                capital += pos * (tp - FEE * 2) / 100
                trades.append('W')
                break
            if l <= sl_price:
                capital += pos * (-sl - FEE * 2) / 100
                trades.append('L')
                break

    if not trades:
        return {'trades': 0, 'wr': 0, 'roi': 0, 'pf': 0}

    w = sum(1 for t in trades if t == 'W')
    roi = (capital / 10000 - 1) * 100
    return {'trades': len(trades), 'wr': w / len(trades) * 100, 'roi': roi}


def train_xgb(X_train, y_train):
    n_neg, n_pos = np.sum(y_train == 0), np.sum(y_train == 1)
    spw = n_neg / max(n_pos, 1)
    model = xgb.XGBClassifier(
        objective='binary:logistic', eval_metric='auc',
        max_depth=6, learning_rate=0.05, n_estimators=200,
        gamma=2, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, scale_pos_weight=spw,
        random_state=42, tree_method='hist', verbosity=0
    )
    model.fit(X_train, y_train, verbose=False)
    return model


def run_validation():
    print("=" * 80)
    print("V12 WALK-FORWARD: V11 vs ATR-only vs ATR+LSTM")
    print("=" * 80)

    all_results = {}

    for crypto in CRYPTOS:
        print(f"\n{'='*80}")
        print(f"  {crypto.upper()}")
        print(f"{'='*80}")

        df_full = load_data(crypto)
        df_full['atr_pct_14'] = calculate_atr_series(df_full, 14)

        atrc = ATR_CONFIG[crypto]
        crypto_results = {}

        for period in PERIODS:
            print(f"\n  --- {period['desc']} ---")

            # ============ V11 BASELINE (fixed TP/SL, no LSTM) ============
            feat_no_lstm = get_features(df_full, crypto, include_lstm=False)

            train_df = df_full[df_full.index < period['xgb_end']]
            test_df = df_full[(df_full.index >= period['test_start']) &
                              (df_full.index < period['test_end'])]

            tc = train_df[train_df['triple_barrier_label'].notna()]
            tc = tc[tc['triple_barrier_label'] != 0]
            X_tr = tc[feat_no_lstm].fillna(0).values
            X_tr = np.nan_to_num(X_tr)
            y_tr = (tc['triple_barrier_label'].values == 1).astype(int)

            model_base = train_xgb(X_tr, y_tr)

            # V11 fixed backtest
            v11_cap = 10000.0
            v11_trades = []
            for i in range(len(test_df) - 1):
                row = test_df.iloc[i]
                feat = row[feat_no_lstm].fillna(0).values.reshape(1, -1)
                feat = np.nan_to_num(feat)
                p = model_base.predict_proba(feat)[0, 1]
                if p < THRESHOLD:
                    continue
                entry = row['close']
                for j in range(1, min(len(test_df) - i, 100)):
                    h, l = test_df.iloc[i + j]['high'], test_df.iloc[i + j]['low']
                    if h >= entry * 1.015:
                        v11_cap += v11_cap * POS_PCT / 100 * (1.5 - FEE * 2) / 100
                        v11_trades.append('W')
                        break
                    if l <= entry * 0.9925:
                        v11_cap += v11_cap * POS_PCT / 100 * (-0.75 - FEE * 2) / 100
                        v11_trades.append('L')
                        break

            v11_roi = (v11_cap / 10000 - 1) * 100
            v11_wr = sum(1 for t in v11_trades if t == 'W') / max(len(v11_trades), 1) * 100

            # ============ V12 ATR-ONLY ============
            r_atr = backtest(model_base, test_df, feat_no_lstm, atrc['tp_mult'], atrc['sl_mult'])

            # ============ V12 ATR+LSTM ============
            print(f"    Training LSTM (cutoff={period['lstm_end']})...")

            try:
                lstm_feats, lstm_model = build_lstm_features_for_crypto(
                    crypto, lstm_train_end=period['lstm_end'],
                    seq_len=20, epochs=30, verbose=False
                )

                # Add LSTM features
                df_lstm = df_full.copy()
                for col in LSTM_COLS:
                    df_lstm[col] = lstm_feats[col].values

                feat_lstm = get_features(df_lstm, crypto, include_lstm=True)

                train_lstm_df = df_lstm[df_lstm.index < period['xgb_end']]
                test_lstm_df = df_lstm[(df_lstm.index >= period['test_start']) &
                                       (df_lstm.index < period['test_end'])]

                tc_l = train_lstm_df[train_lstm_df['triple_barrier_label'].notna()]
                tc_l = tc_l[tc_l['triple_barrier_label'] != 0]
                X_tr_l = tc_l[feat_lstm].fillna(0).values
                X_tr_l = np.nan_to_num(X_tr_l)
                y_tr_l = (tc_l['triple_barrier_label'].values == 1).astype(int)

                model_lstm = train_xgb(X_tr_l, y_tr_l)
                r_lstm = backtest(model_lstm, test_lstm_df, feat_lstm, atrc['tp_mult'], atrc['sl_mult'])

                # LSTM feature importance
                importances = model_lstm.feature_importances_
                lstm_imp = {}
                for col in LSTM_COLS:
                    if col in feat_lstm:
                        idx = feat_lstm.index(col)
                        lstm_imp[col] = float(importances[idx])

            except Exception as e:
                print(f"    LSTM failed: {e}")
                r_lstm = {'trades': 0, 'wr': 0, 'roi': 0}
                lstm_imp = {}

            # Print comparison
            print(f"\n    {'Method':<15} {'Trades':>7} {'WR':>7} {'ROI':>9}")
            print(f"    {'-'*40}")
            print(f"    {'V11 Fixed':<15} {len(v11_trades):>7} {v11_wr:>6.1f}% {v11_roi:>+8.2f}%")
            print(f"    {'V12 ATR':<15} {r_atr['trades']:>7} {r_atr['wr']:>6.1f}% {r_atr['roi']:>+8.2f}%")
            print(f"    {'V12 ATR+LSTM':<15} {r_lstm['trades']:>7} {r_lstm['wr']:>6.1f}% {r_lstm['roi']:>+8.2f}%")

            if lstm_imp:
                top_lstm = sorted(lstm_imp.items(), key=lambda x: -x[1])
                print(f"    LSTM top: {top_lstm[0][0]}={top_lstm[0][1]:.4f}")

            crypto_results[period['name']] = {
                'v11': {'trades': len(v11_trades), 'wr': v11_wr, 'roi': v11_roi},
                'v12_atr': r_atr,
                'v12_lstm': r_lstm,
                'lstm_importance': lstm_imp
            }

        all_results[crypto] = crypto_results

    # ========================================================================
    # SUMMARY TABLE
    # ========================================================================
    print(f"\n\n{'='*80}")
    print("FULL COMPARISON TABLE")
    print("=" * 80)

    print(f"\n{'Crypto':<7} {'Period':<5} {'V11 ROI':>9} {'ATR ROI':>9} {'LSTM ROI':>10} {'LSTM vs V11':>12} {'LSTM vs ATR':>12}")
    print("-" * 70)

    for crypto in CRYPTOS:
        for pname in ['P1', 'P2']:
            if pname not in all_results.get(crypto, {}):
                continue
            r = all_results[crypto][pname]
            v11 = r['v11']['roi']
            atr = r['v12_atr']['roi']
            lstm = r['v12_lstm']['roi']
            print(f"{crypto.upper():<7} {pname:<5} {v11:>+8.2f}% {atr:>+8.2f}% {lstm:>+9.2f}% "
                  f"{lstm - v11:>+11.2f}% {lstm - atr:>+11.2f}%")

    # Consistency
    print(f"\n{'='*80}")
    print("CONSISTENCY ANALYSIS")
    print("=" * 80)

    for crypto in CRYPTOS:
        cr = all_results.get(crypto, {})
        rois_v11 = [cr[p]['v11']['roi'] for p in ['P1', 'P2'] if p in cr]
        rois_atr = [cr[p]['v12_atr']['roi'] for p in ['P1', 'P2'] if p in cr]
        rois_lstm = [cr[p]['v12_lstm']['roi'] for p in ['P1', 'P2'] if p in cr]

        def consist(rois):
            if len(rois) < 2 or np.mean(rois) == 0:
                return 999
            return np.std(rois) / abs(np.mean(rois)) * 100

        c_v11 = consist(rois_v11)
        c_atr = consist(rois_atr)
        c_lstm = consist(rois_lstm)

        print(f"\n  {crypto.upper()}: V11={c_v11:.1f}% | ATR={c_atr:.1f}% | LSTM={c_lstm:.1f}%")
        print(f"    V11  avg ROI: {np.mean(rois_v11):+.2f}%")
        print(f"    ATR  avg ROI: {np.mean(rois_atr):+.2f}%")
        print(f"    LSTM avg ROI: {np.mean(rois_lstm):+.2f}%")

    # Save
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)

    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj

    with open(results_dir / 'walk_forward_lstm_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=convert)

    print(f"\n\nSaved to {results_dir / 'walk_forward_lstm_results.json'}")
    print("=" * 80)
    print("WALK-FORWARD LSTM VALIDATION COMPLETE!")
    print("=" * 80)

    return all_results


if __name__ == '__main__':
    run_validation()
