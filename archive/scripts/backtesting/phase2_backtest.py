"""
PHASE 1 BACKTEST - Test des Optimisations Quick Wins
======================================================
Compare BASELINE vs PHASE 1 OPTIMIZED

BASELINE (V11 TEMPORAL actuel):
- Threshold = 0.5 pour tous
- Toutes les features

PHASE 1 OPTIMIZED:
- BTC: feature-selected model (50 features) + threshold optimal
- ETH: baseline model (348 features) + threshold optimal
- SOL: feature-selected model (50 features) + threshold optimal
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import json


def load_optimal_thresholds():
    """Load optimal thresholds from optimization results"""
    results_dir = Path(__file__).parent.parent / 'optimization' / 'results'

    optimal_thresholds = {}

    for crypto in ['btc', 'eth', 'sol']:
        # Use baseline model thresholds (we don't have optimized models from Optuna yet)
        threshold_file = results_dir / f'{crypto}_baseline_optimal_threshold.json'

        if threshold_file.exists():
            with open(threshold_file) as f:
                data = json.load(f)
                # Use optimal by ROI (better for trading)
                optimal_thresholds[crypto] = data['optimal_by_roi']['threshold']
        else:
            # Fallback to 0.5 if optimization not found
            optimal_thresholds[crypto] = 0.5

    return optimal_thresholds


def simulate_trading(crypto: str, use_phase1: bool, initial_capital=1000.0,
                     tp_pct=1.5, sl_pct=0.75):
    """
    Simulate trading with BASELINE or PHASE 1 configuration

    Args:
        crypto: 'btc', 'eth', or 'sol'
        use_phase1: True = Phase 1 optimized, False = Baseline
        initial_capital: Starting capital ($)
        tp_pct: Take profit (%)
        sl_pct: Stop loss (%)

    Returns:
        Results dictionary and trades dataframe
    """

    # Load model (feature-selected if available and use_phase1=True)
    if use_phase1 and crypto in ['btc', 'sol']:
        model_file = Path(__file__).parent.parent / 'models' / f'{crypto}_v11_feature_selected_top50.joblib'
        model_name = 'feature_selected'
    else:
        model_file = Path(__file__).parent.parent / 'models' / f'{crypto}_v11_classifier.joblib'
        model_name = 'baseline'

    if not model_file.exists():
        print(f"  WARNING: Model not found: {model_file}, using baseline")
        model_file = Path(__file__).parent.parent / 'models' / f'{crypto}_v11_classifier.joblib'
        model_name = 'baseline'

    model = joblib.load(model_file)

    # Load optimal thresholds
    optimal_thresholds = load_optimal_thresholds()

    # Choose threshold
    if use_phase1:
        prob_threshold = optimal_thresholds.get(crypto, 0.5)
        config_name = f'Phase1 (T={prob_threshold:.2f}, {model_name})'
    else:
        prob_threshold = 0.5
        config_name = f'Baseline (T=0.50, all_features)'

    # Load data
    cache_file = Path(__file__).parent.parent / 'data' / 'cache' / f'{crypto}_multi_tf_merged.csv'
    df = pd.read_csv(cache_file, index_col=0, parse_dates=True)

    # Prepare features
    exclude_cols = ['open', 'high', 'low', 'close', 'volume',
                   'label_class', 'label_numeric', 'price_target_pct',
                   'future_price', 'triple_barrier_label']
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    # TEMPORAL SPLIT: Test on 2025+
    df_clean = df[df['triple_barrier_label'].notna()].copy()
    timestamps = df_clean.index
    test_mask = timestamps >= '2025-01-01'
    df_test = df_clean[test_mask].copy()

    # Prepare features
    X_test = df_test[feature_cols].fillna(0).values
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    # Handle feature-selected models (only use top 50 features)
    if model_name == 'feature_selected':
        # Load feature names from results
        results_file = Path(__file__).parent.parent / 'optimization' / 'results' / f'{crypto}_selected_features_top50.json'
        with open(results_file) as f:
            feature_data = json.load(f)
            selected_features = feature_data['selected_feature_names']

        # Get indices of selected features
        feature_indices = [feature_cols.index(f) for f in selected_features if f in feature_cols]
        X_test = X_test[:, feature_indices]

    # Predict P(TP)
    prob_tp = model.predict_proba(X_test)[:, 1]
    df_test['prob_tp'] = prob_tp
    df_test['signal'] = (prob_tp > prob_threshold).astype(int)

    # Simulate trades with capital
    capital = initial_capital
    trades = []
    in_position = False
    entry_idx = None
    entry_price = None
    position_size_usd = None

    max_capital = initial_capital
    min_capital = initial_capital

    for idx in range(len(df_test)):
        row = df_test.iloc[idx]

        # Check entry signal
        if not in_position and row['signal'] == 1 and capital > 0:
            in_position = True
            entry_idx = idx
            entry_price = row['close']
            position_size_usd = capital  # All-in
            continue

        # Check exit
        if in_position:
            current_price = row['close']
            pnl_pct = ((current_price - entry_price) / entry_price) * 100

            # TP hit
            if pnl_pct >= tp_pct:
                profit_usd = position_size_usd * (tp_pct / 100)
                capital += profit_usd

                trades.append({
                    'entry_idx': entry_idx,
                    'exit_idx': idx,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'pnl_pct': tp_pct,
                    'pnl_usd': profit_usd,
                    'capital_after': capital,
                    'outcome': 'TP',
                    'bars_held': idx - entry_idx
                })

                in_position = False
                max_capital = max(max_capital, capital)
                min_capital = min(min_capital, capital)
                continue

            # SL hit
            if pnl_pct <= -sl_pct:
                loss_usd = position_size_usd * (sl_pct / 100)
                capital -= loss_usd

                trades.append({
                    'entry_idx': entry_idx,
                    'exit_idx': idx,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'pnl_pct': -sl_pct,
                    'pnl_usd': -loss_usd,
                    'capital_after': capital,
                    'outcome': 'SL',
                    'bars_held': idx - entry_idx
                })

                in_position = False
                max_capital = max(max_capital, capital)
                min_capital = min(min_capital, capital)
                continue

    # Close any open position
    if in_position:
        final_price = df_test.iloc[-1]['close']
        final_pnl_pct = ((final_price - entry_price) / entry_price) * 100
        final_pnl_usd = position_size_usd * (final_pnl_pct / 100)
        capital += final_pnl_usd

        trades.append({
            'entry_idx': entry_idx,
            'exit_idx': len(df_test) - 1,
            'entry_price': entry_price,
            'exit_price': final_price,
            'pnl_pct': final_pnl_pct,
            'pnl_usd': final_pnl_usd,
            'capital_after': capital,
            'outcome': 'OPEN',
            'bars_held': len(df_test) - 1 - entry_idx
        })

        max_capital = max(max_capital, capital)
        min_capital = min(min_capital, capital)

    # Analyze trades
    if len(trades) == 0:
        return {
            'crypto': crypto.upper(),
            'config': config_name,
            'threshold': prob_threshold,
            'model': model_name,
            'initial_capital': initial_capital,
            'final_capital': capital,
            'total_trades': 0,
            'message': 'No trades executed'
        }, None

    trades_df = pd.DataFrame(trades)

    tp_trades = trades_df[trades_df['outcome'] == 'TP']
    sl_trades = trades_df[trades_df['outcome'] == 'SL']

    # Calculate metrics
    results = {
        'crypto': crypto.upper(),
        'config': config_name,
        'threshold': prob_threshold,
        'model': model_name,
        'initial_capital': initial_capital,
        'final_capital': capital,
        'net_profit': capital - initial_capital,
        'roi_pct': ((capital - initial_capital) / initial_capital) * 100,

        'total_trades': len(trades_df),
        'tp_trades': len(tp_trades),
        'sl_trades': len(sl_trades),
        'win_rate': len(tp_trades) / len(trades_df) * 100 if len(trades_df) > 0 else 0,

        'total_profit_usd': tp_trades['pnl_usd'].sum() if len(tp_trades) > 0 else 0,
        'total_loss_usd': abs(sl_trades['pnl_usd'].sum()) if len(sl_trades) > 0 else 0,

        'max_capital': max_capital,
        'min_capital': min_capital,
        'max_drawdown_usd': initial_capital - min_capital,
        'max_drawdown_pct': ((initial_capital - min_capital) / initial_capital) * 100,

        'avg_bars_held': trades_df['bars_held'].mean(),
    }

    return results, trades_df


def run_phase1_backtest():
    """Run complete Phase 1 backtest comparison"""

    print("="*120)
    print("PHASE 1 BACKTEST - BASELINE vs OPTIMIZED")
    print("="*120)
    print("\nTesting Period: 2025-01-01 onwards (Walk-Forward Validation)")
    print("Capital Initial: $1000 per crypto")
    print("\nConfigurations:")
    print("  BASELINE: Threshold=0.50, All features")
    print("  PHASE 1:  Optimal threshold per crypto, Feature selection (BTC/SOL only)")

    all_results = []

    for crypto in ['btc', 'eth', 'sol']:
        print(f"\n\n{'='*120}")
        print(f"TESTING {crypto.upper()}")
        print('='*120)

        # Run baseline
        print(f"\n  [1/2] Running BASELINE...")
        baseline_results, baseline_trades = simulate_trading(crypto, use_phase1=False)
        all_results.append(baseline_results)

        if baseline_results['total_trades'] > 0:
            print(f"    Trades: {baseline_results['total_trades']}")
            print(f"    Win Rate: {baseline_results['win_rate']:.1f}%")
            print(f"    ROI: {baseline_results['roi_pct']:+.2f}%")
            print(f"    Final Capital: ${baseline_results['final_capital']:.2f}")
        else:
            print(f"    No trades executed")

        # Run Phase 1
        print(f"\n  [2/2] Running PHASE 1 OPTIMIZED...")
        phase1_results, phase1_trades = simulate_trading(crypto, use_phase1=True)
        all_results.append(phase1_results)

        if phase1_results['total_trades'] > 0:
            print(f"    Trades: {phase1_results['total_trades']}")
            print(f"    Win Rate: {phase1_results['win_rate']:.1f}%")
            print(f"    ROI: {phase1_results['roi_pct']:+.2f}%")
            print(f"    Final Capital: ${phase1_results['final_capital']:.2f}")
        else:
            print(f"    No trades executed")

        # Comparison
        if baseline_results['total_trades'] > 0 and phase1_results['total_trades'] > 0:
            roi_diff = phase1_results['roi_pct'] - baseline_results['roi_pct']
            trades_diff = phase1_results['total_trades'] - baseline_results['total_trades']
            wr_diff = phase1_results['win_rate'] - baseline_results['win_rate']

            print(f"\n  IMPROVEMENT:")
            print(f"    ROI: {roi_diff:+.2f}% ({baseline_results['roi_pct']:.1f}% -> {phase1_results['roi_pct']:.1f}%)")
            print(f"    Win Rate: {wr_diff:+.1f}% ({baseline_results['win_rate']:.1f}% -> {phase1_results['win_rate']:.1f}%)")
            print(f"    Trades: {trades_diff:+d} ({baseline_results['total_trades']} -> {phase1_results['total_trades']})")

            if roi_diff > 0:
                print(f"    [OK] Phase 1 is BETTER by {roi_diff:.2f}% ROI!")
            elif roi_diff < 0:
                print(f"    [!!] Phase 1 is WORSE by {abs(roi_diff):.2f}% ROI")
            else:
                print(f"    [--] No change in ROI")

    # Summary Table
    print(f"\n\n{'='*120}")
    print("SUMMARY TABLE - BASELINE vs PHASE 1")
    print('='*120)

    print(f"\n{'Crypto':<8} {'Config':<35} {'Threshold':<10} {'Trades':<8} {'TP':<6} {'SL':<6} "
          f"{'WR%':<8} {'ROI%':<10} {'Capital':<12}")
    print("-"*120)

    for r in all_results:
        if r['total_trades'] == 0:
            print(f"{r['crypto']:<8} {r['config']:<35} {r['threshold']:<10.2f} "
                  f"{'0':<8} {'-':<6} {'-':<6} {'-':<8} {'-':<10} {'-':<12}")
        else:
            print(f"{r['crypto']:<8} {r['config']:<35} {r['threshold']:<10.2f} "
                  f"{r['total_trades']:<8} {r['tp_trades']:<6} {r['sl_trades']:<6} "
                  f"{r['win_rate']:<7.1f}% {r['roi_pct']:<9.2f}% ${r['final_capital']:<11.2f}")

    # Calculate improvements
    print(f"\n\n{'='*120}")
    print("PHASE 1 IMPROVEMENTS")
    print('='*120)

    for crypto in ['btc', 'eth', 'sol']:
        baseline = [r for r in all_results if r['crypto'] == crypto.upper() and 'Baseline' in r['config']]
        phase1 = [r for r in all_results if r['crypto'] == crypto.upper() and 'Phase1' in r['config']]

        if baseline and phase1 and baseline[0]['total_trades'] > 0 and phase1[0]['total_trades'] > 0:
            b = baseline[0]
            p = phase1[0]

            roi_improvement = p['roi_pct'] - b['roi_pct']
            profit_improvement = p['net_profit'] - b['net_profit']
            wr_improvement = p['win_rate'] - b['win_rate']

            print(f"\n{crypto.upper()}:")
            print(f"  ROI: {b['roi_pct']:.2f}% -> {p['roi_pct']:.2f}% ({roi_improvement:+.2f}%)")
            print(f"  Profit: ${b['net_profit']:.2f} -> ${p['net_profit']:.2f} (${profit_improvement:+.2f})")
            print(f"  Win Rate: {b['win_rate']:.1f}% -> {p['win_rate']:.1f}% ({wr_improvement:+.1f}%)")
            print(f"  Trades: {b['total_trades']} -> {p['total_trades']} ({p['total_trades'] - b['total_trades']:+d})")

            if roi_improvement > 5:
                status = "[!!!] EXCELLENT"
            elif roi_improvement > 0:
                status = "[OK] IMPROVED"
            elif roi_improvement > -2:
                status = "[--] NEUTRAL"
            else:
                status = "[!!] DEGRADED"

            print(f"  Status: {status}")

    # Portfolio comparison
    print(f"\n\n{'='*120}")
    print("PORTFOLIO COMPARISON (1000$ per crypto)")
    print('='*120)

    baseline_portfolio = [r for r in all_results if 'Baseline' in r['config'] and r['total_trades'] > 0]
    phase1_portfolio = [r for r in all_results if 'Phase1' in r['config'] and r['total_trades'] > 0]

    if baseline_portfolio:
        baseline_total_initial = sum(r['initial_capital'] for r in baseline_portfolio)
        baseline_total_final = sum(r['final_capital'] for r in baseline_portfolio)
        baseline_total_profit = baseline_total_final - baseline_total_initial
        baseline_roi = (baseline_total_profit / baseline_total_initial) * 100

        print(f"\nBASELINE Portfolio:")
        print(f"  Initial: ${baseline_total_initial:,.2f}")
        print(f"  Final: ${baseline_total_final:,.2f}")
        print(f"  Profit: ${baseline_total_profit:+,.2f}")
        print(f"  ROI: {baseline_roi:+.2f}%")

    if phase1_portfolio:
        phase1_total_initial = sum(r['initial_capital'] for r in phase1_portfolio)
        phase1_total_final = sum(r['final_capital'] for r in phase1_portfolio)
        phase1_total_profit = phase1_total_final - phase1_total_initial
        phase1_roi = (phase1_total_profit / phase1_total_initial) * 100

        print(f"\nPHASE 1 Portfolio:")
        print(f"  Initial: ${phase1_total_initial:,.2f}")
        print(f"  Final: ${phase1_total_final:,.2f}")
        print(f"  Profit: ${phase1_total_profit:+,.2f}")
        print(f"  ROI: {phase1_roi:+.2f}%")

        if baseline_portfolio:
            portfolio_improvement = phase1_roi - baseline_roi
            profit_diff = phase1_total_profit - baseline_total_profit

            print(f"\nPORTFOLIO IMPROVEMENT:")
            print(f"  ROI: {portfolio_improvement:+.2f}% ({baseline_roi:.1f}% -> {phase1_roi:.1f}%)")
            print(f"  Profit: ${profit_diff:+,.2f} (${baseline_total_profit:.2f} -> ${phase1_total_profit:.2f})")

            if portfolio_improvement > 0:
                print(f"  [OK] Phase 1 portfolio is BETTER!")
            else:
                print(f"  [!!] Baseline portfolio was better")

    # Save results
    results_dir = Path(__file__).parent.parent / 'backtesting' / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)

    output_file = results_dir / 'phase1_backtest_results.json'

    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n\n  Résultats sauvegardés: {output_file}")

    print(f"\n{'='*120}")
    print("PHASE 1 BACKTEST COMPLETE!")
    print('='*120)

    print("\n\nNext Steps:")
    print("  [OK] Phase 1 tested")
    print("  [..] Analyze results above")
    print("  [..] Decide: Proceed with Phase 2 (Optuna) or Phase 3 (Data Balancing)?")

    return all_results


if __name__ == '__main__':
    results = run_phase1_backtest()
