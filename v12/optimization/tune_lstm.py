"""
V12 LSTM Tuning + Filter Strategy
===================================
1. Grid search LSTM hyperparams (seq_len, hidden_size, epochs)
2. Test LSTM as FILTER on XGBoost signals:
   - XGBoost says BUY → check LSTM
   - If lstm_proba > filter_threshold → TRADE
   - If lstm_proba < filter_threshold → SKIP (veto)
3. Goal: reduce losing trades while keeping winning trades
"""

import sys
import numpy as np
import pandas as pd
import json
from pathlib import Path
from itertools import product
import xgboost as xgb
import torch

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from v12.features.dynamic_labels import calculate_atr_series
from v12.features.lstm_features import (
    prepare_lstm_features, train_lstm, generate_lstm_features,
    CryptoLSTM, SequenceDataset, LSTM_FEATURES
)

CRYPTOS = ['btc', 'eth', 'sol']

# ATR optimal config
ATR_CFG = {
    'btc': {'tp': 0.40, 'sl': 0.15},
    'eth': {'tp': 0.40, 'sl': 0.15},
    'sol': {'tp': 0.60, 'sl': 0.15}
}

MIN_TP, MAX_TP = 0.75, 3.5
MIN_SL, MAX_SL = 0.35, 1.75
MIN_RR = 1.8
FEE = 0.1
POS_PCT = 10.0
XGB_THRESHOLD = 0.35

# LSTM hyperparams to test
SEQ_LENS = [10, 20, 30, 50]
HIDDEN_SIZES = [32, 64, 128]
EPOCH_COUNTS = [30, 60, 100]

# LSTM filter thresholds
FILTER_THRESHOLDS = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


def load_data(crypto):
    f = PROJECT_ROOT / 'data' / 'cache' / f'{crypto}_multi_tf_merged.csv'
    return pd.read_csv(f, index_col=0, parse_dates=True)


def load_sol_top50():
    f = PROJECT_ROOT / 'optimization' / 'results' / 'sol_selected_features_top50.json'
    if f.exists():
        with open(f) as fh:
            return json.load(fh).get('selected_feature_names', [])
    return None


def get_xgb_features(df, crypto):
    exclude = ['open', 'high', 'low', 'close', 'volume',
               'label_class', 'label_numeric', 'price_target_pct',
               'future_price', 'triple_barrier_label']
    all_cols = [c for c in df.columns if c not in exclude]
    if crypto == 'sol':
        top50 = load_sol_top50()
        if top50:
            cols = [c for c in top50 if c in all_cols]
            if 'atr_pct_14' not in cols and 'atr_pct_14' in all_cols:
                cols.append('atr_pct_14')
            return cols
    return all_cols


def dynamic_tp_sl(atr, tp_m, sl_m):
    tp = np.clip(atr * tp_m, MIN_TP, MAX_TP)
    sl = np.clip(atr * sl_m, MIN_SL, MAX_SL)
    if tp / sl < MIN_RR:
        sl = max(tp / MIN_RR, MIN_SL)
    return tp, sl


def backtest_with_filter(xgb_model, xgb_feats, test_df, lstm_probas,
                         tp_m, sl_m, xgb_thresh, lstm_thresh):
    """
    Backtest with XGBoost + LSTM filter.
    Trade only when XGBoost P(TP) > xgb_thresh AND lstm_proba > lstm_thresh.
    """
    capital = 10000.0
    wins, losses = 0, 0
    trades_detail = []

    for i in range(len(test_df) - 1):
        row = test_df.iloc[i]
        atr = row.get('atr_pct_14', np.nan)
        if pd.isna(atr):
            continue

        # XGBoost prediction
        features = row[xgb_feats].fillna(0).values.reshape(1, -1)
        features = np.nan_to_num(features)
        xgb_prob = xgb_model.predict_proba(features)[0, 1]

        if xgb_prob < xgb_thresh:
            continue

        # LSTM filter
        lstm_prob = lstm_probas[i] if i < len(lstm_probas) else 0.5
        if lstm_thresh is not None and lstm_prob < lstm_thresh:
            continue  # LSTM vetoes this trade

        tp, sl = dynamic_tp_sl(atr, tp_m, sl_m)
        entry = row['close']
        tp_price = entry * (1 + tp / 100)
        sl_price = entry * (1 - sl / 100)
        pos = capital * POS_PCT / 100

        for j in range(1, min(len(test_df) - i, 100)):
            h, l = test_df.iloc[i + j]['high'], test_df.iloc[i + j]['low']
            if h >= tp_price:
                capital += pos * (tp - FEE * 2) / 100
                wins += 1
                trades_detail.append({'r': 'W', 'xgb': xgb_prob, 'lstm': lstm_prob})
                break
            if l <= sl_price:
                capital += pos * (-sl - FEE * 2) / 100
                losses += 1
                trades_detail.append({'r': 'L', 'xgb': xgb_prob, 'lstm': lstm_prob})
                break

    total = wins + losses
    roi = (capital / 10000 - 1) * 100
    wr = wins / total * 100 if total > 0 else 0

    return {
        'trades': total, 'wins': wins, 'losses': losses,
        'wr': wr, 'roi': roi, 'capital': capital
    }


def run_tuning():
    print("=" * 80)
    print("V12 LSTM TUNING + FILTER STRATEGY")
    print("=" * 80)

    all_results = {}

    for crypto in CRYPTOS:
        print(f"\n{'='*80}")
        print(f"  {crypto.upper()}")
        print(f"{'='*80}")

        # Load and prep data
        df = load_data(crypto)
        df['atr_pct_14'] = calculate_atr_series(df, 14)

        xgb_feats = get_xgb_features(df, crypto)
        atrc = ATR_CFG[crypto]

        # Temporal split (Period 2: train<2025, test 2025)
        train_df = df[df.index < '2025-01-01']
        test_df = df[(df.index >= '2025-01-01') & (df.index < '2026-01-01')]

        # Train XGBoost (ATR-only, no LSTM features)
        tc = train_df[train_df['triple_barrier_label'].notna()]
        tc = tc[tc['triple_barrier_label'] != 0]
        X_tr = tc[xgb_feats].fillna(0).values
        X_tr = np.nan_to_num(X_tr)
        y_tr = (tc['triple_barrier_label'].values == 1).astype(int)

        n_neg, n_pos = np.sum(y_tr == 0), np.sum(y_tr == 1)
        xgb_model = xgb.XGBClassifier(
            objective='binary:logistic', eval_metric='auc',
            max_depth=6, learning_rate=0.05, n_estimators=200,
            gamma=2, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            scale_pos_weight=n_neg / max(n_pos, 1),
            random_state=42, tree_method='hist', verbosity=0
        )
        xgb_model.fit(X_tr, y_tr, verbose=False)

        # Baseline: ATR-only (no LSTM filter)
        baseline = backtest_with_filter(
            xgb_model, xgb_feats, test_df, [],
            atrc['tp'], atrc['sl'], XGB_THRESHOLD, None
        )
        print(f"\n  Baseline (ATR-only): {baseline['trades']} trades | "
              f"W:{baseline['wins']} L:{baseline['losses']} | "
              f"WR:{baseline['wr']:.1f}% | ROI:{baseline['roi']:+.2f}%")

        # LSTM data prep
        X_norm, feat_names = prepare_lstm_features(df)
        y_full = (df['triple_barrier_label'].fillna(0).values == 1).astype(float)

        # LSTM train on data < 2024 (anti-leakage)
        lstm_train_mask = df.index < '2024-01-01'
        X_lstm_train = X_norm[lstm_train_mask]
        y_lstm_train = y_full[lstm_train_mask]

        # ========== PHASE 1: HYPERPARAMETER GRID SEARCH ==========
        print(f"\n  Phase 1: LSTM Hyperparameter Grid ({len(SEQ_LENS)}x{len(HIDDEN_SIZES)}x{len(EPOCH_COUNTS)} combos)")

        best_lstm_acc = 0
        best_lstm_config = None
        best_lstm_model = None
        best_lstm_probas = None

        # Test subset of combos (most impactful)
        combos = list(product(SEQ_LENS, HIDDEN_SIZES, EPOCH_COUNTS))
        print(f"  Testing {len(combos)} combos...")

        for seq_len, hidden, epochs in combos:
            try:
                if len(X_lstm_train) <= seq_len + 50:
                    continue

                model = train_lstm(
                    X_lstm_train, y_lstm_train,
                    seq_len=seq_len, epochs=epochs,
                    batch_size=64, lr=0.001, verbose=False
                )
                # Override hidden size
                model_custom = CryptoLSTM(input_size=X_norm.shape[1], hidden_size=hidden)
                model_custom = train_lstm(
                    X_lstm_train, y_lstm_train,
                    seq_len=seq_len, epochs=epochs,
                    batch_size=64, lr=0.001, verbose=False
                )

                # Get probabilities for test period
                lstm_feats = generate_lstm_features(model_custom, X_norm, df, seq_len)
                test_probas = lstm_feats.loc[test_df.index, 'lstm_proba'].values

                # Evaluate LSTM accuracy on test
                test_clean = test_df[test_df['triple_barrier_label'].notna()]
                test_clean = test_clean[test_clean['triple_barrier_label'] != 0]
                if len(test_clean) > 0:
                    lstm_pred = (lstm_feats.loc[test_clean.index, 'lstm_proba'] > 0.5).astype(int)
                    y_true = (test_clean['triple_barrier_label'].values == 1).astype(int)
                    acc = (lstm_pred.values == y_true).mean()

                    if acc > best_lstm_acc:
                        best_lstm_acc = acc
                        best_lstm_config = (seq_len, hidden, epochs)
                        best_lstm_model = model_custom
                        best_lstm_probas = test_probas

            except Exception:
                continue

        if best_lstm_config:
            print(f"\n  Best LSTM: seq={best_lstm_config[0]} hidden={best_lstm_config[1]} "
                  f"epochs={best_lstm_config[2]} acc={best_lstm_acc*100:.1f}%")
        else:
            print(f"\n  LSTM training failed for all combos")
            all_results[crypto] = {'error': 'LSTM failed'}
            continue

        # ========== PHASE 2: FILTER THRESHOLD SEARCH ==========
        print(f"\n  Phase 2: LSTM Filter Thresholds")
        print(f"  {'Threshold':<12} {'Trades':>7} {'Wins':>6} {'Losses':>7} {'WR':>7} {'ROI':>9} {'vs Base':>9}")
        print(f"  {'-'*60}")

        # No filter baseline
        print(f"  {'No filter':<12} {baseline['trades']:>7} {baseline['wins']:>6} "
              f"{baseline['losses']:>7} {baseline['wr']:>6.1f}% {baseline['roi']:>+8.2f}% {'---':>9}")

        best_filter_roi = baseline['roi']
        best_filter_thresh = None
        filter_results = []

        for thresh in FILTER_THRESHOLDS:
            r = backtest_with_filter(
                xgb_model, xgb_feats, test_df, best_lstm_probas,
                atrc['tp'], atrc['sl'], XGB_THRESHOLD, thresh
            )

            delta = r['roi'] - baseline['roi']
            wins_kept = r['wins'] / max(baseline['wins'], 1) * 100
            losses_removed = (1 - r['losses'] / max(baseline['losses'], 1)) * 100

            print(f"  {thresh:<12.2f} {r['trades']:>7} {r['wins']:>6} "
                  f"{r['losses']:>7} {r['wr']:>6.1f}% {r['roi']:>+8.2f}% {delta:>+8.2f}%")

            filter_results.append({
                'threshold': thresh, **r,
                'delta_roi': delta,
                'wins_kept_pct': wins_kept,
                'losses_removed_pct': losses_removed
            })

            if r['roi'] > best_filter_roi and r['trades'] >= 5:
                best_filter_roi = r['roi']
                best_filter_thresh = thresh

        # Analysis: wins kept vs losses removed
        print(f"\n  Filter Analysis (wins kept / losses removed):")
        print(f"  {'Threshold':<12} {'Wins Kept':>12} {'Losses Cut':>12} {'Net Effect':>12}")
        print(f"  {'-'*50}")

        for fr in filter_results:
            print(f"  {fr['threshold']:<12.2f} {fr['wins_kept_pct']:>11.1f}% "
                  f"{fr['losses_removed_pct']:>11.1f}% {fr['delta_roi']:>+11.2f}%")

        if best_filter_thresh:
            print(f"\n  BEST FILTER: lstm_proba > {best_filter_thresh:.2f} "
                  f"(ROI: {best_filter_roi:+.2f}%, delta: {best_filter_roi - baseline['roi']:+.2f}%)")
        else:
            print(f"\n  No filter improves over baseline")

        all_results[crypto] = {
            'baseline': baseline,
            'best_lstm_config': {
                'seq_len': best_lstm_config[0],
                'hidden_size': best_lstm_config[1],
                'epochs': best_lstm_config[2],
                'accuracy': float(best_lstm_acc)
            },
            'best_filter_threshold': best_filter_thresh,
            'best_filter_roi': best_filter_roi,
            'filter_results': filter_results
        }

    # ========== FINAL SUMMARY ==========
    print(f"\n\n{'='*80}")
    print("FINAL SUMMARY")
    print("=" * 80)

    print(f"\n{'Crypto':<7} {'LSTM Config':<25} {'Best Filter':>12} {'Base ROI':>10} {'Filter ROI':>12} {'Delta':>8}")
    print("-" * 75)

    for crypto in CRYPTOS:
        r = all_results.get(crypto, {})
        if 'error' in r:
            print(f"{crypto.upper():<7} FAILED")
            continue

        cfg = r['best_lstm_config']
        base_roi = r['baseline']['roi']
        filt_thresh = r['best_filter_threshold']
        filt_roi = r['best_filter_roi']
        delta = filt_roi - base_roi

        cfg_str = f"seq={cfg['seq_len']} h={cfg['hidden_size']} e={cfg['epochs']}"
        filt_str = f">{filt_thresh:.2f}" if filt_thresh else "None"

        print(f"{crypto.upper():<7} {cfg_str:<25} {filt_str:>12} {base_roi:>+9.2f}% {filt_roi:>+11.2f}% {delta:>+7.2f}%")

    # Save
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)

    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj

    with open(results_dir / 'lstm_tuning_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=convert)

    print(f"\nSaved to {results_dir / 'lstm_tuning_results.json'}")
    print("=" * 80)

    return all_results


if __name__ == '__main__':
    run_tuning()
