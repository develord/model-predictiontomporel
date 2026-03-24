"""
V12 FINAL COMPARISON - V11 vs V12 Production
==============================================
Detailed side-by-side with all metrics:
- Budget initial/final, ROI
- Wins/Losses, Win Rate
- Best/Worst trade in $
- Avg win/loss, Profit Factor, Max Drawdown
- Walk-forward P1+P2

V12 Production = ATR dynamic + Platt Calibration (best combo found)
"""

import sys
import numpy as np
import pandas as pd
import json
from pathlib import Path
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from v12.features.dynamic_labels import calculate_atr_series

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
INITIAL_CAPITAL = 10000.0

PERIODS = [
    {'name': 'P1', 'train_end': '2024-01-01', 'test_start': '2024-01-01',
     'test_end': '2025-01-01', 'desc': 'Test 2024'},
    {'name': 'P2', 'train_end': '2025-01-01', 'test_start': '2025-01-01',
     'test_end': '2026-01-01', 'desc': 'Test 2025'}
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
    if tp / sl < MIN_RR:
        sl = max(tp / MIN_RR, MIN_SL)
    return tp, sl


def full_backtest(model, feat_cols, test_df, mode='v12', tp_m=0, sl_m=0):
    """Full backtest returning detailed trade list."""
    capital = INITIAL_CAPITAL
    trades = []
    peak = capital

    for i in range(len(test_df) - 1):
        row = test_df.iloc[i]
        features = row[feat_cols].fillna(0).values.reshape(1, -1)
        features = np.nan_to_num(features)
        prob = model.predict_proba(features)[0, 1]

        if prob < THRESHOLD:
            continue

        entry = row['close']
        pos = capital * POS_PCT / 100

        if mode == 'v11':
            tp_pct, sl_pct = 1.5, 0.75
        else:
            atr = row.get('atr_pct_14', 4.0)
            tp_pct, sl_pct = dynamic_tp_sl(atr, tp_m, sl_m)

        tp_price = entry * (1 + tp_pct / 100)
        sl_price = entry * (1 - sl_pct / 100)

        for j in range(1, min(len(test_df) - i, 100)):
            h, l = test_df.iloc[i + j]['high'], test_df.iloc[i + j]['low']

            if h >= tp_price:
                pnl_pct = tp_pct - FEE * 2
                pnl = pos * pnl_pct / 100
                capital += pnl
                if capital > peak:
                    peak = capital
                trades.append({
                    'date': str(test_df.index[i].date()),
                    'result': 'WIN', 'pnl': pnl, 'pnl_pct': pnl_pct,
                    'capital': capital, 'prob': prob,
                    'tp_pct': tp_pct, 'sl_pct': sl_pct, 'hold': j
                })
                break

            if l <= sl_price:
                pnl_pct = -sl_pct - FEE * 2
                pnl = pos * pnl_pct / 100
                capital += pnl
                dd = (peak - capital) / peak * 100
                trades.append({
                    'date': str(test_df.index[i].date()),
                    'result': 'LOSE', 'pnl': pnl, 'pnl_pct': pnl_pct,
                    'capital': capital, 'prob': prob,
                    'tp_pct': tp_pct, 'sl_pct': sl_pct, 'hold': j
                })
                break

    return trades, capital


def compute_metrics(trades, final_capital):
    if not trades:
        return {}

    df = pd.DataFrame(trades)
    wins = df[df['result'] == 'WIN']
    losses = df[df['result'] == 'LOSE']

    # Max drawdown
    peak = INITIAL_CAPITAL
    max_dd = 0
    for t in trades:
        if t['capital'] > peak:
            peak = t['capital']
        dd = (peak - t['capital']) / peak * 100
        if dd > max_dd:
            max_dd = dd

    total_win_pnl = wins['pnl'].sum() if len(wins) > 0 else 0
    total_loss_pnl = abs(losses['pnl'].sum()) if len(losses) > 0 else 0

    return {
        'initial': INITIAL_CAPITAL,
        'final': round(final_capital, 2),
        'roi': round((final_capital / INITIAL_CAPITAL - 1) * 100, 2),
        'total_trades': len(df),
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': round(len(wins) / len(df) * 100, 1),
        'best_trade': round(df['pnl'].max(), 2),
        'worst_trade': round(df['pnl'].min(), 2),
        'avg_win': round(wins['pnl'].mean(), 2) if len(wins) > 0 else 0,
        'avg_loss': round(losses['pnl'].mean(), 2) if len(losses) > 0 else 0,
        'profit_factor': round(total_win_pnl / total_loss_pnl, 2) if total_loss_pnl > 0 else 999,
        'max_drawdown': round(max_dd, 2),
        'avg_hold': round(df['hold'].mean(), 1),
        'avg_confidence': round(df['prob'].mean() * 100, 1)
    }


def run():
    print("=" * 100)
    print("V12 FINAL COMPARISON - V11 vs V12 PRODUCTION")
    print("=" * 100)
    print(f"Initial Capital: ${INITIAL_CAPITAL:,.0f} per crypto")
    print(f"V11: Fixed TP=1.5% SL=0.75%")
    print(f"V12: Dynamic ATR TP/SL + Platt Calibration")

    all_results = {}

    for crypto in CRYPTOS:
        print(f"\n{'='*100}")
        print(f"  {crypto.upper()}")
        print(f"{'='*100}")

        df = load_data(crypto)
        df['atr_pct_14'] = calculate_atr_series(df, 14)
        feat_cols = get_features(df, crypto)
        atrc = ATR_CFG[crypto]

        crypto_results = {}

        for period in PERIODS:
            print(f"\n  --- {period['desc']} ---")

            train_df = df[df.index < period['train_end']]
            test_df = df[(df.index >= period['test_start']) & (df.index < period['test_end'])]

            tc = train_df[train_df['triple_barrier_label'].notna()]
            tc = tc[tc['triple_barrier_label'] != 0]
            y_all = (tc['triple_barrier_label'].values == 1).astype(int)

            # 80/20 split for calibration
            split = int(len(tc) * 0.8)
            tc_train, tc_cal = tc.iloc[:split], tc.iloc[split:]
            y_train, y_cal = y_all[:split], y_all[split:]

            X_train = tc_train[feat_cols].fillna(0).values
            X_train = np.nan_to_num(X_train)
            X_cal = tc_cal[feat_cols].fillna(0).values
            X_cal = np.nan_to_num(X_cal)

            n_neg, n_pos = np.sum(y_train == 0), np.sum(y_train == 1)

            # Train base XGBoost
            base_model = xgb.XGBClassifier(
                objective='binary:logistic', eval_metric='auc',
                max_depth=6, learning_rate=0.05, n_estimators=200,
                gamma=2, subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=1.0,
                scale_pos_weight=n_neg / max(n_pos, 1),
                random_state=42, tree_method='hist', verbosity=0
            )
            base_model.fit(X_train, y_train, verbose=False)

            # Calibrated model (Platt)
            cal_model = CalibratedClassifierCV(estimator=base_model, method='sigmoid', cv=2)
            cal_model.fit(X_cal, y_cal)

            # V11 backtest
            v11_trades, v11_cap = full_backtest(base_model, feat_cols, test_df, mode='v11')
            v11_m = compute_metrics(v11_trades, v11_cap)

            # V12 ATR-only
            atr_trades, atr_cap = full_backtest(base_model, feat_cols, test_df, mode='v12',
                                                 tp_m=atrc['tp'], sl_m=atrc['sl'])
            atr_m = compute_metrics(atr_trades, atr_cap)

            # V12 ATR + Platt (PRODUCTION)
            prod_trades, prod_cap = full_backtest(cal_model, feat_cols, test_df, mode='v12',
                                                   tp_m=atrc['tp'], sl_m=atrc['sl'])
            prod_m = compute_metrics(prod_trades, prod_cap)

            crypto_results[period['name']] = {
                'v11': v11_m, 'v12_atr': atr_m, 'v12_prod': prod_m
            }

            # Print detailed comparison
            for label, m in [('V11 Fixed', v11_m), ('V12 ATR', atr_m), ('V12 PROD', prod_m)]:
                print(f"\n    {label}:")
                print(f"      Budget: ${m['initial']:,.0f} → ${m['final']:,.2f} ({m['roi']:+.2f}%)")
                print(f"      Trades: {m['total_trades']} | Wins: {m['wins']} | Losses: {m['losses']} | WR: {m['win_rate']}%")
                print(f"      Best trade: ${m['best_trade']:+.2f} | Worst trade: ${m['worst_trade']:+.2f}")
                print(f"      Avg win: ${m['avg_win']:+.2f} | Avg loss: ${m['avg_loss']:+.2f}")
                print(f"      Profit Factor: {m['profit_factor']} | Max DD: {m['max_drawdown']:.2f}%")
                print(f"      Avg hold: {m['avg_hold']}d | Avg confidence: {m['avg_confidence']}%")

        all_results[crypto] = crypto_results

    # ========== GRAND SUMMARY TABLE ==========
    print(f"\n\n{'='*100}")
    print("GRAND SUMMARY - ALL CRYPTOS, ALL PERIODS")
    print("=" * 100)

    for method_label, method_key in [('V11 FIXED', 'v11'), ('V12 ATR', 'v12_atr'), ('V12 PRODUCTION', 'v12_prod')]:
        print(f"\n{'─'*100}")
        print(f"  {method_label}")
        print(f"{'─'*100}")
        print(f"  {'Crypto':<7} {'Period':<10} {'Budget':>14} {'ROI':>8} {'Trades':>7} {'W':>5} {'L':>5} "
              f"{'WR':>6} {'Best$':>9} {'Worst$':>9} {'AvgW$':>8} {'AvgL$':>8} {'PF':>6} {'MaxDD':>7}")
        print(f"  {'-'*95}")

        total_final = 0
        total_initial = 0

        for crypto in CRYPTOS:
            for pname, pdesc in [('P1', '2024'), ('P2', '2025')]:
                m = all_results.get(crypto, {}).get(pname, {}).get(method_key, {})
                if not m:
                    continue
                total_final += m['final']
                total_initial += m['initial']

                print(f"  {crypto.upper():<7} {pdesc:<10} "
                      f"${m['initial']:>6,.0f}→${m['final']:>7,.0f} "
                      f"{m['roi']:>+7.2f}% "
                      f"{m['total_trades']:>7} {m['wins']:>5} {m['losses']:>5} "
                      f"{m['win_rate']:>5.1f}% "
                      f"${m['best_trade']:>+8.2f} ${m['worst_trade']:>+8.2f} "
                      f"${m['avg_win']:>+7.2f} ${m['avg_loss']:>+7.2f} "
                      f"{m['profit_factor']:>5.2f} {m['max_drawdown']:>6.2f}%")

        port_roi = (total_final / total_initial - 1) * 100
        print(f"\n  {'PORTFOLIO':<7} {'TOTAL':<10} "
              f"${total_initial:>6,.0f}→${total_final:>7,.0f} "
              f"{port_roi:>+7.2f}%")

    # ========== PORTFOLIO SUMMARY ==========
    print(f"\n\n{'='*100}")
    print("PORTFOLIO SUMMARY (3 cryptos x 2 periods = 6 backtests)")
    print("=" * 100)

    print(f"\n  {'Method':<20} {'Total Invested':>15} {'Total Final':>13} {'Avg ROI':>10} {'Total Trades':>13} {'Avg WR':>8}")
    print(f"  {'-'*80}")

    for method_label, method_key in [('V11 FIXED', 'v11'), ('V12 ATR', 'v12_atr'), ('V12 PRODUCTION', 'v12_prod')]:
        total_init = 0
        total_final = 0
        total_trades = 0
        total_wins = 0
        rois = []

        for crypto in CRYPTOS:
            for pname in ['P1', 'P2']:
                m = all_results.get(crypto, {}).get(pname, {}).get(method_key, {})
                if m:
                    total_init += m['initial']
                    total_final += m['final']
                    total_trades += m['total_trades']
                    total_wins += m['wins']
                    rois.append(m['roi'])

        avg_roi = np.mean(rois) if rois else 0
        avg_wr = total_wins / total_trades * 100 if total_trades > 0 else 0

        print(f"  {method_label:<20} ${total_init:>13,.0f} ${total_final:>12,.2f} "
              f"{avg_roi:>+9.2f}% {total_trades:>13} {avg_wr:>7.1f}%")

    # Improvement
    print(f"\n  V12 PRODUCTION vs V11:")
    for crypto in CRYPTOS:
        v11_rois = []
        v12_rois = []
        for pname in ['P1', 'P2']:
            v11 = all_results.get(crypto, {}).get(pname, {}).get('v11', {})
            v12 = all_results.get(crypto, {}).get(pname, {}).get('v12_prod', {})
            if v11 and v12:
                v11_rois.append(v11['roi'])
                v12_rois.append(v12['roi'])
        if v11_rois and v12_rois:
            delta = np.mean(v12_rois) - np.mean(v11_rois)
            print(f"    {crypto.upper()}: V11 avg {np.mean(v11_rois):+.2f}% → V12 avg {np.mean(v12_rois):+.2f}% (delta: {delta:+.2f}%)")

    # Save
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)

    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        return obj

    with open(results_dir / 'final_comparison.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=convert)

    print(f"\nSaved to {results_dir / 'final_comparison.json'}")
    print("=" * 100)

    return all_results


if __name__ == '__main__':
    run()
