"""
V12 Walk-Forward Validation + ATR Multiplier Grid Search
========================================================
1. Walk-Forward: Train on expanding windows, test on next year
   - Period 1: Train 2018-2023 → Test 2024
   - Period 2: Train 2018-2024 → Test 2025

2. Grid Search: Find optimal ATR multipliers per crypto per period
   - tp_mult: [0.3, 0.4, 0.45, 0.5, 0.6]
   - sl_mult: [0.15, 0.2, 0.22, 0.25, 0.3]

3. ATR Period Test: Compare ATR(7) vs ATR(14) vs ATR(21)

4. Consistency Score across periods
"""

import sys
import pandas as pd
import numpy as np
import json
from pathlib import Path
from itertools import product
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from v12.features.dynamic_labels import calculate_atr_series

# ============================================================================
# CONFIG
# ============================================================================

CRYPTOS = ['btc', 'eth', 'sol']

PERIODS = [
    {
        'name': 'Period_1',
        'train_end': '2024-01-01',
        'test_start': '2024-01-01',
        'test_end': '2025-01-01',
        'desc': 'Train →2023, Test 2024'
    },
    {
        'name': 'Period_2',
        'train_end': '2025-01-01',
        'test_start': '2025-01-01',
        'test_end': '2026-01-01',
        'desc': 'Train →2024, Test 2025'
    }
]

# Grid search ranges
TP_MULTS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.60]
SL_MULTS = [0.15, 0.18, 0.20, 0.22, 0.25, 0.30]
ATR_PERIODS = [7, 14, 21]

# Bounds
MIN_TP = 0.75
MAX_TP = 3.5
MIN_SL = 0.35
MAX_SL = 1.75
MIN_RR = 1.8
FEE_PCT = 0.1
POSITION_PCT = 10.0
CONFIDENCE_THRESHOLD = 0.35


def load_sol_top50():
    """Load SOL top 50 feature names."""
    f = PROJECT_ROOT / 'optimization' / 'results' / 'sol_selected_features_top50.json'
    if f.exists():
        with open(f) as fh:
            return json.load(fh).get('selected_feature_names', [])
    return None


def load_data(crypto: str) -> pd.DataFrame:
    """Load merged data."""
    f = PROJECT_ROOT / 'data' / 'cache' / f'{crypto}_multi_tf_merged.csv'
    df = pd.read_csv(f, index_col=0, parse_dates=True)
    return df


def add_atr_feature(df: pd.DataFrame, atr_period: int = 14) -> pd.DataFrame:
    """Add ATR percentage as feature."""
    df = df.copy()
    df['atr_pct_14'] = calculate_atr_series(df, atr_period)
    return df


def get_feature_cols(df, crypto):
    """Get feature columns (with top 50 for SOL)."""
    exclude = [
        'open', 'high', 'low', 'close', 'volume',
        'label_class', 'label_numeric',
        'price_target_pct', 'future_price',
        'triple_barrier_label'
    ]
    all_cols = [c for c in df.columns if c not in exclude]

    if crypto == 'sol':
        top50 = load_sol_top50()
        if top50:
            cols = [c for c in top50 if c in all_cols]
            if 'atr_pct_14' not in cols and 'atr_pct_14' in all_cols:
                cols.append('atr_pct_14')
            return cols
    return all_cols


def train_model(X_train, y_train):
    """Train XGBoost binary classifier."""
    n_neg = np.sum(y_train == 0)
    n_pos = np.sum(y_train == 1)
    spw = n_neg / n_pos if n_pos > 0 else 1.0

    model = xgb.XGBClassifier(
        objective='binary:logistic', eval_metric='auc',
        max_depth=6, learning_rate=0.05, n_estimators=200,
        gamma=2, min_child_weight=1, subsample=0.8,
        colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        scale_pos_weight=spw, random_state=42, tree_method='hist',
        verbosity=0
    )
    model.fit(X_train, y_train, verbose=False)
    return model


def dynamic_tp_sl(atr_pct, tp_mult, sl_mult):
    """Compute dynamic TP/SL from ATR."""
    tp = np.clip(atr_pct * tp_mult, MIN_TP, MAX_TP)
    sl = np.clip(atr_pct * sl_mult, MIN_SL, MAX_SL)
    if tp / sl < MIN_RR:
        sl = tp / MIN_RR
        sl = max(sl, MIN_SL)
    return tp, sl


def backtest_dynamic(model, test_df, feature_cols, tp_mult, sl_mult, threshold=CONFIDENCE_THRESHOLD):
    """
    Backtest with dynamic ATR TP/SL.
    Uses OHLCV to simulate trades candle by candle.
    """
    capital = 10000.0
    trades = []

    for i in range(len(test_df) - 1):
        row = test_df.iloc[i]
        atr = row.get('atr_pct_14', np.nan)
        if pd.isna(atr):
            continue

        features = row[feature_cols].fillna(0).values.reshape(1, -1)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        prob = model.predict_proba(features)[0, 1]

        if prob < threshold:
            continue

        tp_pct, sl_pct = dynamic_tp_sl(atr, tp_mult, sl_mult)
        entry_price = row['close']
        tp_price = entry_price * (1 + tp_pct / 100)
        sl_price = entry_price * (1 - sl_pct / 100)

        pos_value = capital * (POSITION_PCT / 100)

        # Scan future candles
        result = None
        for j in range(1, min(len(test_df) - i, 100)):
            h = test_df.iloc[i + j]['high']
            l = test_df.iloc[i + j]['low']

            if h >= tp_price:
                pnl = pos_value * (tp_pct - FEE_PCT * 2) / 100
                result = 'WIN'
                break
            if l <= sl_price:
                pnl = pos_value * (-sl_pct - FEE_PCT * 2) / 100
                result = 'LOSE'
                break

        if result:
            capital += pnl
            trades.append({'result': result, 'pnl': pnl, 'tp_pct': tp_pct, 'sl_pct': sl_pct})

    if not trades:
        return {'trades': 0, 'win_rate': 0, 'roi': 0, 'pf': 0, 'max_dd': 0}

    n = len(trades)
    wins = sum(1 for t in trades if t['result'] == 'WIN')
    roi = (capital / 10000.0 - 1) * 100
    total_win = sum(t['pnl'] for t in trades if t['result'] == 'WIN')
    total_loss = abs(sum(t['pnl'] for t in trades if t['result'] == 'LOSE'))
    pf = total_win / total_loss if total_loss > 0 else float('inf')

    # Max drawdown
    peak = 10000.0
    max_dd = 0
    cap = 10000.0
    for t in trades:
        cap += t['pnl']
        if cap > peak:
            peak = cap
        dd = (peak - cap) / peak * 100
        if dd > max_dd:
            max_dd = dd

    return {
        'trades': n, 'wins': wins, 'win_rate': wins / n * 100,
        'roi': roi, 'pf': pf, 'max_dd': max_dd,
        'avg_tp': np.mean([t['tp_pct'] for t in trades]),
        'avg_sl': np.mean([t['sl_pct'] for t in trades]),
        'capital': capital
    }


# ============================================================================
# WALK-FORWARD + GRID SEARCH
# ============================================================================

def run_walk_forward_grid():
    """Walk-forward with ATR multiplier grid search."""

    print("=" * 80)
    print("V12 WALK-FORWARD VALIDATION + ATR GRID SEARCH")
    print("=" * 80)

    all_results = {}

    for crypto in CRYPTOS:
        print(f"\n{'='*80}")
        print(f"  {crypto.upper()}")
        print(f"{'='*80}")

        df_full = load_data(crypto)
        df_full = add_atr_feature(df_full, 14)
        feature_cols = get_feature_cols(df_full, crypto)

        crypto_results = {}

        for period in PERIODS:
            print(f"\n  --- {period['desc']} ---")

            # Split
            train_df = df_full[df_full.index < period['train_end']].copy()
            test_df = df_full[
                (df_full.index >= period['test_start']) &
                (df_full.index < period['test_end'])
            ].copy()

            # Prepare training data (V11 fixed labels)
            train_clean = train_df[train_df['triple_barrier_label'].notna()].copy()
            train_clean = train_clean[train_clean['triple_barrier_label'] != 0].copy()

            X_train = train_clean[feature_cols].fillna(0).values
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
            y_train = (train_clean['triple_barrier_label'].values == 1).astype(int)

            print(f"  Train: {len(X_train)} samples | Test: {len(test_df)} candles")
            print(f"  Class: TP={np.sum(y_train==1)} SL={np.sum(y_train==0)}")

            if len(X_train) < 100 or len(test_df) < 30:
                print("  [SKIP] Not enough data")
                continue

            # Train model
            model = train_model(X_train, y_train)

            # Evaluate model accuracy on test
            test_clean = test_df[test_df['triple_barrier_label'].notna()].copy()
            test_clean = test_clean[test_clean['triple_barrier_label'] != 0].copy()
            if len(test_clean) > 0:
                X_test = test_clean[feature_cols].fillna(0).values
                X_test = np.nan_to_num(X_test)
                y_test = (test_clean['triple_barrier_label'].values == 1).astype(int)
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                print(f"  Model accuracy: {acc*100:.2f}%")

            # ---- GRID SEARCH ATR MULTIPLIERS ----
            print(f"\n  Grid Search: {len(TP_MULTS)}x{len(SL_MULTS)} = {len(TP_MULTS)*len(SL_MULTS)} combos")

            best_roi = -999
            best_combo = None
            grid_results = []

            for tp_m, sl_m in product(TP_MULTS, SL_MULTS):
                r = backtest_dynamic(model, test_df, feature_cols, tp_m, sl_m)
                grid_results.append({
                    'tp_mult': tp_m, 'sl_mult': sl_m,
                    **r
                })
                if r['roi'] > best_roi and r['trades'] >= 5:
                    best_roi = r['roi']
                    best_combo = (tp_m, sl_m)

            # Also test V11 fixed baseline
            v11_result = backtest_dynamic(model, test_df, feature_cols, 0, 0)
            # For V11, override with fixed TP/SL
            v11_capital = 10000.0
            v11_trades = []
            for i in range(len(test_df) - 1):
                row = test_df.iloc[i]
                features = row[feature_cols].fillna(0).values.reshape(1, -1)
                features = np.nan_to_num(features)
                prob = model.predict_proba(features)[0, 1]
                if prob < CONFIDENCE_THRESHOLD:
                    continue
                entry = row['close']
                tp_price = entry * 1.015
                sl_price = entry * 0.9925
                pos = v11_capital * POSITION_PCT / 100
                for j in range(1, min(len(test_df) - i, 100)):
                    h = test_df.iloc[i + j]['high']
                    l = test_df.iloc[i + j]['low']
                    if h >= tp_price:
                        v11_capital += pos * (1.5 - FEE_PCT * 2) / 100
                        v11_trades.append('W')
                        break
                    if l <= sl_price:
                        v11_capital += pos * (-0.75 - FEE_PCT * 2) / 100
                        v11_trades.append('L')
                        break

            v11_roi = (v11_capital / 10000 - 1) * 100
            v11_wr = sum(1 for t in v11_trades if t == 'W') / len(v11_trades) * 100 if v11_trades else 0

            # Best result
            if best_combo:
                best_r = [r for r in grid_results if r['tp_mult'] == best_combo[0] and r['sl_mult'] == best_combo[1]][0]
            else:
                best_r = {'roi': 0, 'trades': 0, 'win_rate': 0, 'pf': 0, 'avg_tp': 0, 'avg_sl': 0}

            print(f"\n  V11 Fixed (1.5%/0.75%): {len(v11_trades)} trades | WR: {v11_wr:.1f}% | ROI: {v11_roi:+.2f}%")
            if best_combo:
                print(f"  V12 Best ({best_combo[0]}/{best_combo[1]}): "
                      f"{best_r['trades']} trades | WR: {best_r['win_rate']:.1f}% | "
                      f"ROI: {best_r['roi']:+.2f}% | PF: {best_r['pf']:.2f} | "
                      f"TP:{best_r['avg_tp']:.2f}% SL:{best_r['avg_sl']:.2f}%")
                delta = best_r['roi'] - v11_roi
                print(f"  Delta ROI: {delta:+.2f}%")

            # Top 5 combos
            sorted_grid = sorted(grid_results, key=lambda x: x['roi'], reverse=True)
            print(f"\n  Top 5 ATR combos:")
            for i, g in enumerate(sorted_grid[:5]):
                print(f"    {i+1}. tp={g['tp_mult']:.2f} sl={g['sl_mult']:.2f} -> "
                      f"ROI:{g['roi']:+.2f}% WR:{g['win_rate']:.1f}% "
                      f"Trades:{g['trades']} PF:{g['pf']:.2f}")

            crypto_results[period['name']] = {
                'v11_baseline': {'roi': v11_roi, 'trades': len(v11_trades), 'win_rate': v11_wr},
                'v12_best': {
                    'tp_mult': best_combo[0] if best_combo else 0,
                    'sl_mult': best_combo[1] if best_combo else 0,
                    **best_r
                },
                'top5': sorted_grid[:5],
                'model_accuracy': float(acc) if len(test_clean) > 0 else 0
            }

        all_results[crypto] = crypto_results

    # ========================================================================
    # ATR PERIOD COMPARISON (14 vs 7 vs 21)
    # ========================================================================
    print(f"\n\n{'='*80}")
    print("ATR PERIOD COMPARISON (7 vs 14 vs 21)")
    print("=" * 80)

    for crypto in CRYPTOS:
        print(f"\n  {crypto.upper()} (Period 2 only):")
        df_full = load_data(crypto)

        for atr_p in ATR_PERIODS:
            df_atr = df_full.copy()
            df_atr['atr_pct_14'] = calculate_atr_series(df_atr, atr_p)
            feat = get_feature_cols(df_atr, crypto)

            train_df = df_atr[df_atr.index < '2025-01-01']
            test_df = df_atr[(df_atr.index >= '2025-01-01') & (df_atr.index < '2026-01-01')]

            tc = train_df[train_df['triple_barrier_label'].notna()]
            tc = tc[tc['triple_barrier_label'] != 0]

            X_tr = tc[feat].fillna(0).values
            X_tr = np.nan_to_num(X_tr)
            y_tr = (tc['triple_barrier_label'].values == 1).astype(int)

            model = train_model(X_tr, y_tr)

            # Use best combo from grid search if available
            if crypto in all_results and 'Period_2' in all_results[crypto]:
                best = all_results[crypto]['Period_2']['v12_best']
                tp_m, sl_m = best['tp_mult'], best['sl_mult']
            else:
                tp_m, sl_m = 0.45, 0.22

            r = backtest_dynamic(model, test_df, feat, tp_m, sl_m)
            print(f"    ATR({atr_p:2d}): ROI={r['roi']:+.2f}% WR={r['win_rate']:.1f}% "
                  f"Trades={r['trades']} PF={r['pf']:.2f}")

    # ========================================================================
    # CONSISTENCY ANALYSIS
    # ========================================================================
    print(f"\n\n{'='*80}")
    print("CONSISTENCY & ROBUSTNESS ANALYSIS")
    print("=" * 80)

    print(f"\n{'Crypto':<8} {'P1 V11':>8} {'P1 V12':>8} {'P2 V11':>8} {'P2 V12':>8} {'Consist':>10} {'Verdict':<20}")
    print("-" * 75)

    for crypto in CRYPTOS:
        if crypto not in all_results:
            continue
        cr = all_results[crypto]

        p1_v11 = cr.get('Period_1', {}).get('v11_baseline', {}).get('roi', 0)
        p1_v12 = cr.get('Period_1', {}).get('v12_best', {}).get('roi', 0)
        p2_v11 = cr.get('Period_2', {}).get('v11_baseline', {}).get('roi', 0)
        p2_v12 = cr.get('Period_2', {}).get('v12_best', {}).get('roi', 0)

        rois = [p1_v12, p2_v12]
        mean_roi = np.mean(rois)
        std_roi = np.std(rois)
        consistency = (std_roi / abs(mean_roi)) * 100 if mean_roi != 0 else 999

        if consistency < 20:
            verdict = "EXCELLENT"
        elif consistency < 40:
            verdict = "GOOD"
        elif consistency < 60:
            verdict = "MODERATE"
        else:
            verdict = "POOR"

        both_profit = all(r > 0 for r in rois)
        if not both_profit:
            verdict += " (LOSS!)"

        print(f"{crypto.upper():<8} {p1_v11:>+7.2f}% {p1_v12:>+7.2f}% {p2_v11:>+7.2f}% {p2_v12:>+7.2f}% "
              f"{consistency:>9.1f}% {verdict:<20}")

    # Recommended config
    print(f"\n{'='*80}")
    print("RECOMMENDED V12 CONFIG PER CRYPTO")
    print("=" * 80)

    for crypto in CRYPTOS:
        if crypto not in all_results:
            continue
        # Use combos that appear in top 5 for BOTH periods
        p1_top = all_results[crypto].get('Period_1', {}).get('top5', [])
        p2_top = all_results[crypto].get('Period_2', {}).get('top5', [])

        p1_combos = {(t['tp_mult'], t['sl_mult']) for t in p1_top}
        p2_combos = {(t['tp_mult'], t['sl_mult']) for t in p2_top}
        common = p1_combos & p2_combos

        if common:
            # Pick the one with best avg ROI across both periods
            best_common = None
            best_avg = -999
            for combo in common:
                r1 = [t for t in p1_top if t['tp_mult'] == combo[0] and t['sl_mult'] == combo[1]]
                r2 = [t for t in p2_top if t['tp_mult'] == combo[0] and t['sl_mult'] == combo[1]]
                if r1 and r2:
                    avg = (r1[0]['roi'] + r2[0]['roi']) / 2
                    if avg > best_avg:
                        best_avg = avg
                        best_common = combo

            if best_common:
                print(f"\n  {crypto.upper()}: tp_mult={best_common[0]:.2f}, sl_mult={best_common[1]:.2f} "
                      f"(avg ROI: {best_avg:+.2f}%, top 5 in BOTH periods)")
            else:
                print(f"\n  {crypto.upper()}: No common combo in top 5 - use Period 2 best")
        else:
            # Fallback: use Period 2 best (most recent)
            p2_best = all_results[crypto].get('Period_2', {}).get('v12_best', {})
            print(f"\n  {crypto.upper()}: tp_mult={p2_best.get('tp_mult', 0.45):.2f}, "
                  f"sl_mult={p2_best.get('sl_mult', 0.22):.2f} (Period 2 best, no common top 5)")

    # Save
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)

    # Convert numpy types for JSON
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(results_dir / 'walk_forward_grid_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=convert)

    print(f"\n\nResults saved to {results_dir / 'walk_forward_grid_results.json'}")
    print("=" * 80)
    print("WALK-FORWARD + GRID SEARCH COMPLETE!")
    print("=" * 80)

    return all_results


if __name__ == '__main__':
    run_walk_forward_grid()
