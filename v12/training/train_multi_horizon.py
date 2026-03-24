"""
V12 Multi-Horizon Convergence
==============================
Train 3 models (3d/5d/7d horizons), trade only when they converge.

Strategy:
- Model_3d: predicts P(TP within 3 days)
- Model_5d: predicts P(TP within 5 days)
- Model_7d: predicts P(TP within 7 days)

Convergence modes:
- "2of3": trade when >=2 models say BUY (majority)
- "3of3": trade when all 3 models say BUY (unanimous)

Combined with dynamic ATR TP/SL at execution.
Walk-forward validated on 2 periods.
"""

import sys
import numpy as np
import pandas as pd
import json
from pathlib import Path
import xgboost as xgb
from sklearn.metrics import accuracy_score

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from v12.features.dynamic_labels import calculate_atr_series
from v12.features.multi_horizon_labels import generate_multi_horizon_labels

CRYPTOS = ['btc', 'eth', 'sol']
HORIZONS = [3, 5, 7]

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
    {'name': 'P1', 'train_end': '2024-01-01', 'test_start': '2024-01-01', 'test_end': '2025-01-01',
     'desc': 'Train <2024, Test 2024'},
    {'name': 'P2', 'train_end': '2025-01-01', 'test_start': '2025-01-01', 'test_end': '2026-01-01',
     'desc': 'Train <2025, Test 2025'}
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


def get_features(df, crypto):
    exclude = ['open', 'high', 'low', 'close', 'volume',
               'label_class', 'label_numeric', 'price_target_pct',
               'future_price', 'triple_barrier_label',
               'label_3d', 'label_5d', 'label_7d']
    all_cols = [c for c in df.columns if c not in exclude]

    if crypto == 'sol':
        top50 = load_sol_top50()
        if top50:
            cols = [c for c in top50 if c in all_cols]
            if 'atr_pct_14' not in cols and 'atr_pct_14' in all_cols:
                cols.append('atr_pct_14')
            return cols
    return all_cols


def train_model(X, y):
    n_neg, n_pos = np.sum(y == 0), np.sum(y == 1)
    spw = n_neg / max(n_pos, 1)
    model = xgb.XGBClassifier(
        objective='binary:logistic', eval_metric='auc',
        max_depth=6, learning_rate=0.05, n_estimators=200,
        gamma=2, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, scale_pos_weight=spw,
        random_state=42, tree_method='hist', verbosity=0
    )
    model.fit(X, y, verbose=False)
    return model


def dynamic_tp_sl(atr, tp_m, sl_m):
    tp = np.clip(atr * tp_m, MIN_TP, MAX_TP)
    sl = np.clip(atr * sl_m, MIN_SL, MAX_SL)
    if tp / sl < MIN_RR:
        sl = max(tp / MIN_RR, MIN_SL)
    return tp, sl


def backtest_convergence(models, feat_cols, test_df, tp_m, sl_m, mode='2of3'):
    """
    Backtest with multi-horizon convergence.
    mode: '2of3' or '3of3' or 'any' (standard single model)
    """
    capital = 10000.0
    wins, losses = 0, 0
    min_agree = {'any': 1, '2of3': 2, '3of3': 3}[mode]

    for i in range(len(test_df) - 1):
        row = test_df.iloc[i]
        atr = row.get('atr_pct_14', np.nan)
        if pd.isna(atr):
            continue

        features = row[feat_cols].fillna(0).values.reshape(1, -1)
        features = np.nan_to_num(features)

        # Get predictions from all models
        buy_votes = 0
        for h, model in models.items():
            prob = model.predict_proba(features)[0, 1]
            if prob >= THRESHOLD:
                buy_votes += 1

        if buy_votes < min_agree:
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


def backtest_v11_fixed(model_7d, feat_cols, test_df):
    """V11 baseline: single 7d model, fixed TP/SL."""
    capital = 10000.0
    wins, losses = 0, 0

    for i in range(len(test_df) - 1):
        row = test_df.iloc[i]
        features = row[feat_cols].fillna(0).values.reshape(1, -1)
        features = np.nan_to_num(features)
        prob = model_7d.predict_proba(features)[0, 1]

        if prob < THRESHOLD:
            continue

        entry = row['close']
        pos = capital * POS_PCT / 100
        for j in range(1, min(len(test_df) - i, 100)):
            h, l = test_df.iloc[i + j]['high'], test_df.iloc[i + j]['low']
            if h >= entry * 1.015:
                capital += pos * (1.5 - FEE * 2) / 100
                wins += 1
                break
            if l <= entry * 0.9925:
                capital += pos * (-0.75 - FEE * 2) / 100
                losses += 1
                break

    total = wins + losses
    roi = (capital / 10000 - 1) * 100
    wr = wins / total * 100 if total > 0 else 0
    return {'trades': total, 'wins': wins, 'losses': losses, 'wr': wr, 'roi': roi}


def run():
    print("=" * 80)
    print("V12 MULTI-HORIZON CONVERGENCE (3d/5d/7d)")
    print("=" * 80)

    all_results = {}

    for crypto in CRYPTOS:
        print(f"\n{'='*80}")
        print(f"  {crypto.upper()}")
        print(f"{'='*80}")

        df = load_data(crypto)
        df['atr_pct_14'] = calculate_atr_series(df, 14)
        df = generate_multi_horizon_labels(df)

        feat_cols = get_features(df, crypto)
        atrc = ATR_CFG[crypto]

        crypto_results = {}

        for period in PERIODS:
            print(f"\n  --- {period['desc']} ---")

            train_df = df[df.index < period['train_end']]
            test_df = df[(df.index >= period['test_start']) & (df.index < period['test_end'])]

            # Train 3 models (one per horizon)
            models = {}
            for horizon in HORIZONS:
                label_col = f'label_{horizon}d'
                tc = train_df[train_df[label_col].notna()]
                tc = tc[tc[label_col] != 0]

                X = tc[feat_cols].fillna(0).values
                X = np.nan_to_num(X)
                y = (tc[label_col].values == 1).astype(int)

                n_tp = np.sum(y == 1)
                n_sl = np.sum(y == 0)

                models[horizon] = train_model(X, y)
                print(f"    Model {horizon}d: {len(X)} samples (TP:{n_tp} SL:{n_sl})")

            # Evaluate accuracy per model
            test_clean = test_df[test_df['label_7d'].notna()]
            test_clean = test_clean[test_clean['label_7d'] != 0]
            if len(test_clean) > 0:
                X_test = test_clean[feat_cols].fillna(0).values
                X_test = np.nan_to_num(X_test)
                for horizon in HORIZONS:
                    lc = f'label_{horizon}d'
                    tc2 = test_clean[test_clean[lc].notna()]
                    tc2 = tc2[tc2[lc] != 0]
                    if len(tc2) > 0:
                        X_t = tc2[feat_cols].fillna(0).values
                        X_t = np.nan_to_num(X_t)
                        y_t = (tc2[lc].values == 1).astype(int)
                        y_p = models[horizon].predict(X_t)
                        acc = accuracy_score(y_t, y_p)
                        print(f"    {horizon}d accuracy: {acc*100:.1f}%")

            # Backtest all modes
            v11 = backtest_v11_fixed(models[7], feat_cols, test_df)
            atr_7d = backtest_convergence(
                {7: models[7]}, feat_cols, test_df, atrc['tp'], atrc['sl'], 'any')
            conv_2of3 = backtest_convergence(
                models, feat_cols, test_df, atrc['tp'], atrc['sl'], '2of3')
            conv_3of3 = backtest_convergence(
                models, feat_cols, test_df, atrc['tp'], atrc['sl'], '3of3')

            # Print comparison
            print(f"\n    {'Method':<20} {'Trades':>7} {'Wins':>6} {'Losses':>7} {'WR':>7} {'ROI':>9}")
            print(f"    {'-'*58}")
            for name, r in [('V11 Fixed 7d', v11), ('ATR 7d only', atr_7d),
                            ('Convergence 2/3', conv_2of3), ('Convergence 3/3', conv_3of3)]:
                print(f"    {name:<20} {r['trades']:>7} {r['wins']:>6} {r['losses']:>7} "
                      f"{r['wr']:>6.1f}% {r['roi']:>+8.2f}%")

            crypto_results[period['name']] = {
                'v11': v11, 'atr_7d': atr_7d,
                'conv_2of3': conv_2of3, 'conv_3of3': conv_3of3
            }

        all_results[crypto] = crypto_results

    # ========== SUMMARY ==========
    print(f"\n\n{'='*80}")
    print("FULL COMPARISON TABLE")
    print("=" * 80)

    print(f"\n{'Crypto':<7} {'Per':<4} {'V11':>9} {'ATR 7d':>9} {'2of3':>9} {'3of3':>9} {'Best':>12}")
    print("-" * 55)

    for crypto in CRYPTOS:
        for pname in ['P1', 'P2']:
            r = all_results.get(crypto, {}).get(pname, {})
            if not r:
                continue
            v11 = r['v11']['roi']
            atr = r['atr_7d']['roi']
            c2 = r['conv_2of3']['roi']
            c3 = r['conv_3of3']['roi']

            best_val = max(v11, atr, c2, c3)
            best_name = {v11: 'V11', atr: 'ATR', c2: '2of3', c3: '3of3'}[best_val]

            print(f"{crypto.upper():<7} {pname:<4} {v11:>+8.2f}% {atr:>+8.2f}% "
                  f"{c2:>+8.2f}% {c3:>+8.2f}% {best_name:>12}")

    # Win rate comparison
    print(f"\n{'Crypto':<7} {'Per':<4} {'V11 WR':>8} {'ATR WR':>8} {'2/3 WR':>8} {'3/3 WR':>8} "
          f"{'V11 T':>6} {'ATR T':>6} {'2/3 T':>6} {'3/3 T':>6}")
    print("-" * 72)

    for crypto in CRYPTOS:
        for pname in ['P1', 'P2']:
            r = all_results.get(crypto, {}).get(pname, {})
            if not r:
                continue
            print(f"{crypto.upper():<7} {pname:<4} "
                  f"{r['v11']['wr']:>7.1f}% {r['atr_7d']['wr']:>7.1f}% "
                  f"{r['conv_2of3']['wr']:>7.1f}% {r['conv_3of3']['wr']:>7.1f}% "
                  f"{r['v11']['trades']:>6} {r['atr_7d']['trades']:>6} "
                  f"{r['conv_2of3']['trades']:>6} {r['conv_3of3']['trades']:>6}")

    # Consistency
    print(f"\n{'='*80}")
    print("CONSISTENCY (avg ROI across P1+P2)")
    print("=" * 80)

    for crypto in CRYPTOS:
        cr = all_results.get(crypto, {})
        for method in ['v11', 'atr_7d', 'conv_2of3', 'conv_3of3']:
            rois = [cr[p][method]['roi'] for p in ['P1', 'P2'] if p in cr]
            if len(rois) == 2:
                avg = np.mean(rois)
                std = np.std(rois)
                cons = std / abs(avg) * 100 if avg != 0 else 999
                print(f"  {crypto.upper()} {method:<12}: avg={avg:+.2f}% consistency={cons:.1f}%")

    # Save
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)

    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        return obj

    with open(results_dir / 'multi_horizon_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=convert)

    print(f"\nSaved to {results_dir / 'multi_horizon_results.json'}")
    print("=" * 80)
    print("MULTI-HORIZON CONVERGENCE COMPLETE!")
    print("=" * 80)

    return all_results


if __name__ == '__main__':
    run()
