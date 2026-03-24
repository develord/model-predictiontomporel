"""
V12 Priority 3+4: Sentiment Features + Probability Calibration
================================================================
P3: Add Fear&Greed, Funding Rates, Open Interest as features
P4: Calibrate XGBoost probabilities with Platt Scaling / Isotonic Regression

Walk-forward validated on 2 periods.
Compare: ATR-only | ATR+Sentiment | ATR+Calibration | ATR+Sentiment+Calibration
"""

import sys
import numpy as np
import pandas as pd
import json
from pathlib import Path
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.calibration import CalibratedClassifierCV

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from v12.features.dynamic_labels import calculate_atr_series
from v12.features.sentiment_features import build_sentiment_features, SENTIMENT_COLS

CRYPTOS = ['btc', 'eth', 'sol']

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
THRESHOLD = 0.35

PERIODS = [
    {'name': 'P1', 'train_end': '2024-01-01', 'test_start': '2024-01-01',
     'test_end': '2025-01-01', 'desc': 'Train <2024, Test 2024'},
    {'name': 'P2', 'train_end': '2025-01-01', 'test_start': '2025-01-01',
     'test_end': '2026-01-01', 'desc': 'Train <2025, Test 2025'}
]


def load_data(crypto):
    f = PROJECT_ROOT / 'data' / 'cache' / f'{crypto}_multi_tf_merged.csv'
    return pd.read_csv(f, index_col=0, parse_dates=True)


def load_sol_top50():
    f = PROJECT_ROOT / 'optimization' / 'results' / 'sol_selected_features_top50.json'
    if f.exists():
        with open(f) as fh:
            return json.load(fh).get('selected_feature_names', [])
    return None


def get_features(df, crypto, include_sentiment=False):
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
            if include_sentiment:
                for sc in SENTIMENT_COLS:
                    if sc in all_cols and sc not in cols:
                        cols.append(sc)
            return cols

    if not include_sentiment:
        return [c for c in all_cols if c not in SENTIMENT_COLS]
    return all_cols


def dynamic_tp_sl(atr, tp_m, sl_m):
    tp = np.clip(atr * tp_m, MIN_TP, MAX_TP)
    sl = np.clip(atr * sl_m, MIN_SL, MAX_SL)
    if tp / sl < MIN_RR:
        sl = max(tp / MIN_RR, MIN_SL)
    return tp, sl


def train_xgb(X, y):
    n_neg, n_pos = np.sum(y == 0), np.sum(y == 1)
    model = xgb.XGBClassifier(
        objective='binary:logistic', eval_metric='auc',
        max_depth=6, learning_rate=0.05, n_estimators=200,
        gamma=2, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        scale_pos_weight=n_neg / max(n_pos, 1),
        random_state=42, tree_method='hist', verbosity=0
    )
    model.fit(X, y, verbose=False)
    return model


def calibrate_model(model, X_cal, y_cal, method='sigmoid'):
    """
    Calibrate probabilities using Platt Scaling (sigmoid) or Isotonic Regression.
    Uses a held-out calibration set from the end of training data.
    sklearn >= 1.8: use estimator= instead of cv='prefit'
    """
    from sklearn.calibration import CalibratedClassifierCV
    try:
        # sklearn >= 1.8
        cal = CalibratedClassifierCV(estimator=model, method=method, cv=2)
        cal.fit(X_cal, y_cal)
    except Exception:
        # Fallback: wrap manually
        cal = CalibratedClassifierCV(model, method=method, cv=2)
        cal.fit(X_cal, y_cal)
    return cal


def backtest(model, feat_cols, test_df, tp_m, sl_m, threshold=THRESHOLD):
    capital = 10000.0
    wins, losses = 0, 0

    for i in range(len(test_df) - 1):
        row = test_df.iloc[i]
        atr = row.get('atr_pct_14', np.nan)
        if pd.isna(atr):
            continue

        features = row[feat_cols].fillna(0).values.reshape(1, -1)
        features = np.nan_to_num(features)
        prob = model.predict_proba(features)[0, 1]

        if prob < threshold:
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
                wins += 1
                break
            if l <= sl_price:
                capital += pos * (-sl - FEE * 2) / 100
                losses += 1
                break

    total = wins + losses
    roi = (capital / 10000 - 1) * 100
    wr = wins / total * 100 if total > 0 else 0
    return {'trades': total, 'wins': wins, 'losses': losses, 'wr': wr, 'roi': roi}


def run():
    print("=" * 80)
    print("V12 PRIORITY 3+4: SENTIMENT + CALIBRATION")
    print("=" * 80)

    all_results = {}

    for crypto in CRYPTOS:
        print(f"\n{'='*80}")
        print(f"  {crypto.upper()}")
        print(f"{'='*80}")

        df = load_data(crypto)
        df['atr_pct_14'] = calculate_atr_series(df, 14)

        # Add sentiment features
        sentiment = build_sentiment_features(crypto, df)
        for col in SENTIMENT_COLS:
            if col in sentiment.columns:
                df[col] = sentiment[col].values

        feat_no_sent = get_features(df, crypto, include_sentiment=False)
        feat_with_sent = get_features(df, crypto, include_sentiment=True)

        atrc = ATR_CFG[crypto]
        crypto_results = {}

        for period in PERIODS:
            print(f"\n  --- {period['desc']} ---")

            train_df = df[df.index < period['train_end']]
            test_df = df[(df.index >= period['test_start']) & (df.index < period['test_end'])]

            # Prepare training data
            tc = train_df[train_df['triple_barrier_label'].notna()]
            tc = tc[tc['triple_barrier_label'] != 0]
            y_tr = (tc['triple_barrier_label'].values == 1).astype(int)

            # Split train into train + calibration (80/20)
            split_idx = int(len(tc) * 0.8)
            tc_train = tc.iloc[:split_idx]
            tc_cal = tc.iloc[split_idx:]
            y_train = y_tr[:split_idx]
            y_cal = y_tr[split_idx:]

            results = {}

            # ===== A. ATR-only (baseline) =====
            X_tr_ns = tc_train[feat_no_sent].fillna(0).values
            X_tr_ns = np.nan_to_num(X_tr_ns)
            X_cal_ns = tc_cal[feat_no_sent].fillna(0).values
            X_cal_ns = np.nan_to_num(X_cal_ns)

            model_base = train_xgb(X_tr_ns, y_train)
            results['ATR-only'] = backtest(model_base, feat_no_sent, test_df,
                                           atrc['tp'], atrc['sl'])

            # ===== B. ATR + Sentiment =====
            X_tr_s = tc_train[feat_with_sent].fillna(0).values
            X_tr_s = np.nan_to_num(X_tr_s)
            X_cal_s = tc_cal[feat_with_sent].fillna(0).values
            X_cal_s = np.nan_to_num(X_cal_s)

            model_sent = train_xgb(X_tr_s, y_train)
            results['ATR+Sent'] = backtest(model_sent, feat_with_sent, test_df,
                                           atrc['tp'], atrc['sl'])

            # Sentiment feature importance
            importances = model_sent.feature_importances_
            sent_imp = {}
            for col in SENTIMENT_COLS:
                if col in feat_with_sent:
                    idx = feat_with_sent.index(col)
                    sent_imp[col] = float(importances[idx])

            # ===== C. ATR + Platt Calibration =====
            model_cal_platt = calibrate_model(model_base, X_cal_ns, y_cal, 'sigmoid')
            results['ATR+Platt'] = backtest(model_cal_platt, feat_no_sent, test_df,
                                            atrc['tp'], atrc['sl'])

            # ===== D. ATR + Isotonic Calibration =====
            model_cal_iso = calibrate_model(model_base, X_cal_ns, y_cal, 'isotonic')
            results['ATR+Isotonic'] = backtest(model_cal_iso, feat_no_sent, test_df,
                                               atrc['tp'], atrc['sl'])

            # ===== E. ATR + Sentiment + Platt =====
            model_sent_cal = calibrate_model(model_sent, X_cal_s, y_cal, 'sigmoid')
            results['ATR+Sent+Platt'] = backtest(model_sent_cal, feat_with_sent, test_df,
                                                  atrc['tp'], atrc['sl'])

            # Print
            print(f"\n    {'Method':<20} {'Trades':>7} {'Wins':>6} {'Losses':>7} {'WR':>7} {'ROI':>9}")
            print(f"    {'-'*58}")
            for name, r in results.items():
                print(f"    {name:<20} {r['trades']:>7} {r['wins']:>6} {r['losses']:>7} "
                      f"{r['wr']:>6.1f}% {r['roi']:>+8.2f}%")

            # Sentiment importance
            if sent_imp:
                top3 = sorted(sent_imp.items(), key=lambda x: -x[1])[:3]
                print(f"\n    Sentiment top 3: ", end="")
                print(" | ".join(f"{k}={v:.4f}" for k, v in top3))

            crypto_results[period['name']] = {
                method: r for method, r in results.items()
            }
            crypto_results[period['name']]['sentiment_importance'] = sent_imp

        all_results[crypto] = crypto_results

    # ========== SUMMARY ==========
    print(f"\n\n{'='*80}")
    print("FULL COMPARISON (ROI)")
    print("=" * 80)

    methods = ['ATR-only', 'ATR+Sent', 'ATR+Platt', 'ATR+Isotonic', 'ATR+Sent+Platt']
    header = f"{'Crypto':<7} {'Per':<4} " + " ".join(f"{m:>14}" for m in methods)
    print(f"\n{header}")
    print("-" * len(header))

    for crypto in CRYPTOS:
        for pname in ['P1', 'P2']:
            cr = all_results.get(crypto, {}).get(pname, {})
            if not cr:
                continue
            line = f"{crypto.upper():<7} {pname:<4} "
            best_roi = -999
            for m in methods:
                r = cr.get(m, {})
                roi = r.get('roi', 0)
                if roi > best_roi:
                    best_roi = roi
                line += f"{roi:>+13.2f}% "
            print(line)

    # Avg ROI + Consistency
    print(f"\n{'='*80}")
    print("AVG ROI & CONSISTENCY")
    print("=" * 80)

    for crypto in CRYPTOS:
        cr = all_results.get(crypto, {})
        print(f"\n  {crypto.upper()}:")
        for m in methods:
            rois = [cr[p][m]['roi'] for p in ['P1', 'P2'] if p in cr and m in cr[p]]
            if len(rois) == 2:
                avg = np.mean(rois)
                cons = np.std(rois) / abs(avg) * 100 if avg != 0 else 999
                print(f"    {m:<20}: avg={avg:>+.2f}% consistency={cons:.1f}%")

    # Save
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)

    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        return obj

    with open(results_dir / 'prio3_4_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=convert)

    print(f"\nSaved to {results_dir / 'prio3_4_results.json'}")
    print("=" * 80)

    return all_results


if __name__ == '__main__':
    run()
