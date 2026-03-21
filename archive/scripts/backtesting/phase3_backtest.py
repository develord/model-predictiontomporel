"""
PHASE 3 BACKTEST - Test SMOTE Models en ROI
============================================
Compare Phase 1 (optimal) vs Phase 3 (SMOTE balanced)

EXPECTED RESULTS:
- Phase 3 degraded accuracy for ETH (-8.78%) and SOL (-3.52%)
- Only BTC improved slightly (+2.96%)
- This backtest confirms if SMOTE also degrades ROI

PHASE 1 RESULTS (baseline):
- BTC: ~22% ROI (threshold 0.37)
- ETH: +45.07% ROI (threshold 0.35)
- SOL: +64.48% ROI (threshold 0.35, feature-selected)
- Portfolio: +43.38% ROI
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from phase1_backtest import simulate_trading

def main():
    print("=" * 70)
    print("PHASE 3: SMOTE MODEL BACKTEST (ALL CRYPTOS)")
    print("=" * 70)
    print()

    results_summary = {
        'btc': {},
        'eth': {},
        'sol': {}
    }

    for crypto in ['btc', 'eth', 'sol']:
        print(f"\n{'#'*70}")
        print(f"# {crypto.upper()}")
        print(f"{'#'*70}")

        # Determine optimal configuration for each crypto
        if crypto == 'btc':
            threshold = 0.37
            use_phase1_features = False  # BTC uses baseline in Phase 1
        elif crypto == 'eth':
            threshold = 0.35
            use_phase1_features = False  # ETH uses baseline in Phase 1
        else:  # sol
            threshold = 0.35
            use_phase1_features = True  # SOL uses feature-selected in Phase 1

        # Test Phase 1 (optimal configuration)
        print(f"\n[1] Phase 1 (Optimal Config, T={threshold}):")
        print("-" * 70)

        phase1_results, _ = simulate_trading(
            crypto=crypto,
            use_phase1=use_phase1_features
        )

        print(f"  Model: {'Feature-selected' if use_phase1_features else 'Baseline'}")
        print(f"  Accuracy: {phase1_results.get('accuracy', 0):.2f}%")
        print(f"  Total Trades: {phase1_results['total_trades']}")
        print(f"  Win Rate: {phase1_results['win_rate']:.2f}%")
        print(f"  ROI: {phase1_results['roi_pct']:+.2f}%")
        print(f"  Capital: ${phase1_results['initial_capital']:.2f} -> ${phase1_results['final_capital']:.2f}")

        results_summary[crypto]['phase1'] = phase1_results

        # Test Phase 3 (SMOTE)
        print(f"\n[2] Phase 3 (SMOTE Balanced, T={threshold}):")
        print("-" * 70)

        # Load Phase 3 model
        import joblib
        import pandas as pd
        import numpy as np
        import json

        models_dir = Path(__file__).parent.parent / 'models'

        if crypto in ['btc', 'sol']:
            model_file = models_dir / f'{crypto}_v11_phase3_smote_features.joblib'
        else:
            model_file = models_dir / f'{crypto}_v11_phase3_smote.joblib'

        if not model_file.exists():
            print(f"  ERROR: Model not found: {model_file}")
            continue

        model = joblib.load(model_file)

        # Load features if needed
        if crypto in ['btc', 'sol']:
            features_file = Path(__file__).parent.parent / 'optimization' / 'results' / f'{crypto}_selected_features_top50.json'
            with open(features_file) as f:
                feature_data = json.load(f)
                selected_features = feature_data['selected_feature_names']
        else:
            selected_features = None

        # Load data
        cache_file = Path(__file__).parent.parent / 'data' / 'cache' / f'{crypto}_multi_tf_merged.csv'
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)

        # Prepare features
        exclude_cols = ['open', 'high', 'low', 'close', 'volume',
                       'label_class', 'label_numeric', 'price_target_pct',
                       'future_price', 'triple_barrier_label']

        if selected_features:
            feature_cols = selected_features
        else:
            feature_cols = [col for col in df.columns if col not in exclude_cols]

        # TEMPORAL SPLIT: Test on 2025+
        df_clean = df[df['triple_barrier_label'].notna()].copy()
        test_mask = df_clean.index >= '2025-01-01'
        df_test = df_clean[test_mask].copy()

        # Prepare test features
        X_test = df_test[feature_cols].fillna(0).values
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
        y_test = df_test['label_numeric'].values

        # Predict
        probs = model.predict_proba(X_test)[:, 1]
        signals = (probs > threshold).astype(int)

        # Backtest
        initial_capital = 1000.0
        tp_pct = 1.5
        sl_pct = 0.75
        capital = initial_capital
        trades = []

        for i, (idx, row) in enumerate(df_test.iterrows()):
            if signals[i] == 1:  # BUY signal
                actual_label = y_test[i]
                prob = probs[i]

                if actual_label == 1:  # TP
                    pnl_pct = tp_pct
                    outcome = 'TP'
                else:  # SL
                    pnl_pct = -sl_pct
                    outcome = 'SL'

                pnl = capital * (pnl_pct / 100)
                capital += pnl

                trades.append({
                    'timestamp': idx,
                    'signal_prob': prob,
                    'outcome': outcome,
                    'pnl_pct': pnl_pct,
                    'pnl_usd': pnl,
                    'capital': capital
                })

        # Calculate metrics
        trades_df = pd.DataFrame(trades)

        if len(trades_df) > 0:
            total_trades = len(trades_df)
            tp_trades = (trades_df['outcome'] == 'TP').sum()
            sl_trades = (trades_df['outcome'] == 'SL').sum()
            win_rate = tp_trades / total_trades if total_trades > 0 else 0

            final_capital = capital
            roi_pct = ((final_capital - initial_capital) / initial_capital) * 100

            # Accuracy
            y_pred = signals
            accuracy = (y_pred == y_test).mean()
        else:
            total_trades = 0
            tp_trades = 0
            sl_trades = 0
            win_rate = 0
            final_capital = initial_capital
            roi_pct = 0
            accuracy = 0

        phase3_results = {
            'initial_capital': initial_capital,
            'final_capital': final_capital,
            'roi_pct': roi_pct,
            'total_trades': total_trades,
            'tp_trades': tp_trades,
            'sl_trades': sl_trades,
            'win_rate': win_rate * 100,
            'accuracy': accuracy * 100,
            'tp_pct': tp_pct,
            'sl_pct': sl_pct
        }

        results_summary[crypto]['phase3'] = phase3_results

        print(f"  Model: SMOTE {'+ Features' if crypto in ['btc', 'sol'] else ''}")
        print(f"  Accuracy: {phase3_results['accuracy']:.2f}%")
        print(f"  Total Trades: {phase3_results['total_trades']}")
        print(f"  Win Rate: {phase3_results['win_rate']:.2f}%")
        print(f"  ROI: {phase3_results['roi_pct']:+.2f}%")
        print(f"  Capital: ${phase3_results['initial_capital']:.2f} -> ${phase3_results['final_capital']:.2f}")

        # Comparison
        print(f"\n[COMPARISON]")
        print("-" * 70)
        roi_diff = phase3_results['roi_pct'] - phase1_results['roi_pct']
        acc_diff = phase3_results['accuracy'] - phase1_results.get('accuracy', 0)
        trades_diff = phase3_results['total_trades'] - phase1_results['total_trades']

        print(f"  Accuracy: {acc_diff:+.2f}% ({phase1_results.get('accuracy', 0):.1f}% -> {phase3_results['accuracy']:.1f}%)")
        print(f"  ROI: {roi_diff:+.2f}% ({phase1_results['roi_pct']:.1f}% -> {phase3_results['roi_pct']:.1f}%)")
        print(f"  Trades: {trades_diff:+d} ({phase1_results['total_trades']} -> {phase3_results['total_trades']})")

        if roi_diff < 0:
            print(f"  VERDICT: Phase 1 IS BETTER! Phase 3 is {-roi_diff:.2f}% worse [X]")
        else:
            print(f"  VERDICT: Phase 3 IS BETTER! +{roi_diff:.2f}% improvement [OK]")

    # Portfolio summary
    print("\n\n" + "=" * 70)
    print("PORTFOLIO SUMMARY")
    print("=" * 70)

    phase1_portfolio = sum([results_summary[c]['phase1']['roi_pct'] for c in ['btc', 'eth', 'sol']]) / 3
    phase3_portfolio = sum([results_summary[c]['phase3']['roi_pct'] for c in ['btc', 'eth', 'sol']]) / 3
    portfolio_diff = phase3_portfolio - phase1_portfolio

    print(f"\nPhase 1 Portfolio ROI: {phase1_portfolio:+.2f}%")
    print(f"Phase 3 Portfolio ROI: {phase3_portfolio:+.2f}%")
    print(f"Difference: {portfolio_diff:+.2f}%")

    print("\n" + "=" * 70)
    print("FINAL RECOMMENDATION")
    print("=" * 70)

    if portfolio_diff < 0:
        print(f"\n[X] SMOTE DEGRADES PERFORMANCE by {-portfolio_diff:.2f}%")
        print(f"\n[OK] KEEP PHASE 1 CONFIGURATION (Optimal)")
    else:
        print(f"\n[OK] SMOTE IMPROVES PERFORMANCE by {portfolio_diff:.2f}%")
        print(f"\n-> Consider using Phase 3 models")

    print()


if __name__ == '__main__':
    main()
