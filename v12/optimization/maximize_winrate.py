"""
V12 Maximize Win Rate
======================
Grid search on:
- Confidence threshold: [0.40 → 0.80]
- TP multiplier: [0.20 → 0.60] (tighter TP = easier to hit)
- SL multiplier: [0.15 → 0.40] (wider SL = fewer stops)

Objective: MAX win rate while ROI > 0 and trades >= 20
Walk-forward on P1+P2.
"""

import sys
import numpy as np
import pandas as pd
import json
from pathlib import Path
from itertools import product
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from v12.features.dynamic_labels import calculate_atr_series

CRYPTOS = ['btc', 'eth', 'sol']

MIN_TP, MAX_TP = 0.50, 3.5
MIN_SL, MAX_SL = 0.35, 2.5
FEE = 0.1
POS_PCT = 10.0

THRESHOLDS = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
TP_MULTS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]
SL_MULTS = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]

PERIODS = [
    {'name': 'P1', 'train_end': '2024-01-01', 'test_start': '2024-01-01', 'test_end': '2025-01-01'},
    {'name': 'P2', 'train_end': '2025-01-01', 'test_start': '2025-01-01', 'test_end': '2026-01-01'}
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
    return tp, sl


def backtest(model, feat_cols, test_df, tp_m, sl_m, threshold):
    capital = 10000.0
    wins, losses = 0, 0
    trade_pnls = []

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
                pnl = pos * (tp - FEE * 2) / 100
                capital += pnl
                wins += 1
                trade_pnls.append(pnl)
                break
            if l <= sl_price:
                pnl = pos * (-sl - FEE * 2) / 100
                capital += pnl
                losses += 1
                trade_pnls.append(pnl)
                break

    total = wins + losses
    roi = (capital / 10000 - 1) * 100
    wr = wins / total * 100 if total > 0 else 0

    best = max(trade_pnls) if trade_pnls else 0
    worst = min(trade_pnls) if trade_pnls else 0
    avg_w = np.mean([p for p in trade_pnls if p > 0]) if wins > 0 else 0
    avg_l = np.mean([p for p in trade_pnls if p < 0]) if losses > 0 else 0

    return {
        'trades': total, 'wins': wins, 'losses': losses, 'wr': wr,
        'roi': roi, 'capital': capital,
        'best': best, 'worst': worst, 'avg_win': avg_w, 'avg_loss': avg_l
    }


def run():
    print("=" * 100)
    print("V12 MAXIMIZE WIN RATE - Grid Search")
    print("=" * 100)

    all_results = {}

    for crypto in CRYPTOS:
        print(f"\n{'='*100}")
        print(f"  {crypto.upper()}")
        print(f"{'='*100}")

        df = load_data(crypto)
        df['atr_pct_14'] = calculate_atr_series(df, 14)
        feat_cols = get_features(df, crypto)

        best_combos = []

        for period in PERIODS:
            print(f"\n  --- {period['name']} ---")

            train_df = df[df.index < period['train_end']]
            test_df = df[(df.index >= period['test_start']) & (df.index < period['test_end'])]

            tc = train_df[train_df['triple_barrier_label'].notna()]
            tc = tc[tc['triple_barrier_label'] != 0]
            y_all = (tc['triple_barrier_label'].values == 1).astype(int)

            split = int(len(tc) * 0.8)
            X_train = tc.iloc[:split][feat_cols].fillna(0).values
            X_train = np.nan_to_num(X_train)
            y_train = y_all[:split]
            X_cal = tc.iloc[split:][feat_cols].fillna(0).values
            X_cal = np.nan_to_num(X_cal)
            y_cal = y_all[split:]

            n_neg, n_pos = np.sum(y_train == 0), np.sum(y_train == 1)
            base = xgb.XGBClassifier(
                objective='binary:logistic', eval_metric='auc',
                max_depth=6, learning_rate=0.05, n_estimators=200,
                gamma=2, subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=1.0,
                scale_pos_weight=n_neg / max(n_pos, 1),
                random_state=42, tree_method='hist', verbosity=0
            )
            base.fit(X_train, y_train, verbose=False)

            model = CalibratedClassifierCV(estimator=base, method='sigmoid', cv=2)
            model.fit(X_cal, y_cal)

            # Grid search
            combos = len(THRESHOLDS) * len(TP_MULTS) * len(SL_MULTS)
            print(f"  Grid: {combos} combos...")

            results = []
            for thresh, tp_m, sl_m in product(THRESHOLDS, TP_MULTS, SL_MULTS):
                r = backtest(model, feat_cols, test_df, tp_m, sl_m, thresh)
                r['threshold'] = thresh
                r['tp_mult'] = tp_m
                r['sl_mult'] = sl_m
                results.append(r)

            # Filter: trades >= 20 and ROI > 0
            valid = [r for r in results if r['trades'] >= 20 and r['roi'] > 0]

            # Sort by win rate (descending), then by ROI
            valid.sort(key=lambda x: (-x['wr'], -x['roi']))

            print(f"\n  Top 10 by WIN RATE (trades>=20, ROI>0):")
            print(f"  {'Thresh':>7} {'TP_m':>5} {'SL_m':>5} {'Trades':>7} {'Wins':>5} {'Losses':>6} "
                  f"{'WR':>7} {'ROI':>8} {'BestTr$':>9} {'WorstTr$':>9} {'AvgW$':>8} {'AvgL$':>8}")
            print(f"  {'-'*90}")

            for r in valid[:10]:
                print(f"  {r['threshold']:>7.2f} {r['tp_mult']:>5.2f} {r['sl_mult']:>5.2f} "
                      f"{r['trades']:>7} {r['wins']:>5} {r['losses']:>6} "
                      f"{r['wr']:>6.1f}% {r['roi']:>+7.2f}% "
                      f"${r['best']:>+8.2f} ${r['worst']:>+8.2f} "
                      f"${r['avg_win']:>+7.2f} ${r['avg_loss']:>+7.2f}")

            if valid:
                best_combos.append(valid[0])

        # Find combo that works in BOTH periods
        if len(best_combos) == 2:
            # Also search for combos in top 20 of both periods
            all_p1 = [r for r in results if r['trades'] >= 20 and r['roi'] > 0]  # last period's results
            # Re-run P1 to get its results too - use best from each
            print(f"\n  {'='*60}")
            print(f"  BEST CONFIG for {crypto.upper()}:")
            for i, bc in enumerate(best_combos):
                print(f"    P{i+1}: thresh={bc['threshold']:.2f} tp={bc['tp_mult']:.2f} sl={bc['sl_mult']:.2f} "
                      f"-> WR={bc['wr']:.1f}% ROI={bc['roi']:+.2f}% "
                      f"Trades={bc['trades']} W={bc['wins']} L={bc['losses']}")

        all_results[crypto] = {
            f'P{i+1}': {
                'best': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                         for k, v in bc.items()}
            }
            for i, bc in enumerate(best_combos)
        }

    # ========== FINAL ==========
    print(f"\n\n{'='*100}")
    print("FINAL: MAX WIN RATE CONFIG PER CRYPTO")
    print("=" * 100)

    print(f"\n{'Crypto':<7} {'Period':<7} {'Thresh':>7} {'TP_m':>5} {'SL_m':>5} "
          f"{'Trades':>7} {'Wins':>5} {'Losses':>6} {'WR':>7} {'ROI':>9} "
          f"{'Best$':>9} {'Worst$':>9} {'AvgW$':>8} {'AvgL$':>8}")
    print("-" * 105)

    for crypto in CRYPTOS:
        cr = all_results.get(crypto, {})
        for pname in ['P1', 'P2']:
            b = cr.get(pname, {}).get('best', {})
            if b:
                print(f"{crypto.upper():<7} {pname:<7} {b.get('threshold',0):>7.2f} "
                      f"{b.get('tp_mult',0):>5.2f} {b.get('sl_mult',0):>5.2f} "
                      f"{b.get('trades',0):>7.0f} {b.get('wins',0):>5.0f} {b.get('losses',0):>6.0f} "
                      f"{b.get('wr',0):>6.1f}% {b.get('roi',0):>+8.2f}% "
                      f"${b.get('best',0):>+8.2f} ${b.get('worst',0):>+8.2f} "
                      f"${b.get('avg_win',0):>+7.2f} ${b.get('avg_loss',0):>+7.2f}")

    # Save
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)

    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        return obj

    with open(results_dir / 'maximize_winrate.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=convert)

    print(f"\nSaved to {results_dir / 'maximize_winrate.json'}")
    print("=" * 100)

    return all_results


if __name__ == '__main__':
    run()
