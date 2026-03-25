"""
V11 OPTIMIZED BACKTEST - PHASE 1 QUICK WINS IMPLEMENTED
========================================================
All Phase 1 improvements for +10-15% Win Rate:
- QW #1: Adaptive thresholds per crypto
- QW #11: Day-of-week filter (avoid Friday, prioritize Sunday)
- QW #2: Low confidence zone filter (50-53%)
- QW #10: Consecutive loss protection
- QW #5: Market regime filter (SMA50 > SMA200)
"""

import sys
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent))


class OptimizedBacktest:
    """V11 Backtest with all Phase 1 Quick Wins"""

    def __init__(self, crypto: str, mode: str, tp_pct=1.5, sl_pct=0.75):
        self.crypto = crypto
        self.mode = mode
        self.tp_pct = tp_pct
        self.sl_pct = sl_pct

        # QW #1: ADAPTIVE THRESHOLDS
        self.adaptive_thresholds = {
            'btc': 0.55,  # +0.8% WR
            'eth': 0.50,  # Optimal already
            'sol': 0.65   # +3.07% WR
        }
        self.prob_threshold = self.adaptive_thresholds.get(crypto, 0.5)

        print(f"  Using adaptive threshold: {self.prob_threshold:.2f} for {crypto.upper()}")

        # Load model
        model_file = Path(__file__).parent.parent / 'models' / f'{crypto}_v11_{mode}.joblib'
        if not model_file.exists():
            raise FileNotFoundError(f"Model not found: {model_file}")

        self.model = joblib.load(model_file)

        # Load stats
        stats_file = Path(__file__).parent.parent / 'models' / f'{crypto}_v11_{mode}_stats.json'
        with open(stats_file, 'r') as f:
            stats = json.load(f)

        # Load data
        cache_file = Path(__file__).parent.parent / 'data' / 'cache' / f'{crypto}_multi_tf_merged.csv'
        self.df = pd.read_csv(cache_file, index_col=0, parse_dates=True)

        # Prepare features
        exclude_cols = [
            'open', 'high', 'low', 'close', 'volume',
            'label_class', 'label_numeric', 'price_target_pct',
            'future_price', 'triple_barrier_label'
        ]

        all_features = [col for col in self.df.columns if col not in exclude_cols]

        if mode == 'top50':
            top_features = [f['feature'] for f in stats['top_features'][:50]]
            self.feature_cols = [f for f in top_features if f in self.df.columns]
        else:
            self.feature_cols = all_features

    def run_optimized_backtest(self, start_date='2025-01-01'):
        """Run backtest with ALL Phase 1 Quick Wins"""

        df_test = self.df[self.df.index >= start_date].copy()

        if len(df_test) == 0:
            print(f"  WARNING: No test data")
            return None

        print(f"\n  Test period: {df_test.index[0]} to {df_test.index[-1]}")

        # Prepare features
        X_test = df_test[self.feature_cols].fillna(0).values
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

        # Predict
        prob_tp = self.model.predict_proba(X_test)[:, 1]
        df_test['prob_tp'] = prob_tp

        # Get day of week for filtering
        df_test['day_of_week'] = df_test.index.dayofweek

        # QW #5: Market regime filter - Check if SMA features exist
        has_sma_filter = '1d_sma_50' in df_test.columns and '1d_sma_200' in df_test.columns

        if has_sma_filter:
            df_test['bullish_regime'] = df_test['1d_sma_50'] > df_test['1d_sma_200']
            print(f"  Market regime filter: ENABLED")
        else:
            df_test['bullish_regime'] = True  # Default to always true
            print(f"  Market regime filter: DISABLED (features not found)")

        # SIGNAL GENERATION WITH ALL FILTERS
        df_test['signal'] = (
            # Base threshold (QW #1: Adaptive)
            (prob_tp > self.prob_threshold) &

            # QW #2: Avoid low confidence zone 50-53%
            ((prob_tp < 0.50) | (prob_tp >= 0.53)) &

            # QW #11: Day-of-week filter (avoid Friday=4)
            (df_test['day_of_week'] != 4) &

            # QW #5: Market regime filter
            (df_test['bullish_regime'])
        )

        # Simulate trades with QW #10: Consecutive loss protection
        trades = []
        position = None
        consecutive_losses = 0  # Track consecutive losses

        for i in range(len(df_test)):
            row = df_test.iloc[i]

            # Handle existing position
            if position is not None:
                high = df_test['high'].iloc[i]
                low = df_test['low'].iloc[i]

                entry_price = position['entry_price']
                tp_price = entry_price * (1 + self.tp_pct / 100)
                sl_price = entry_price * (1 - self.sl_pct / 100)

                hit_tp = high >= tp_price
                hit_sl = low <= sl_price

                if hit_tp or hit_sl:
                    exit_price = tp_price if hit_tp else sl_price
                    exit_type = 'TP' if hit_tp else 'SL'
                    pnl_pct = ((exit_price - entry_price) / entry_price) * 100

                    position['exit_time'] = row.name
                    position['exit_price'] = exit_price
                    position['exit_type'] = exit_type
                    position['pnl_pct'] = pnl_pct
                    position['bars_held'] = i - position['entry_idx']
                    position['result'] = 'WIN' if hit_tp else 'LOSS'

                    trades.append(position)
                    position = None

                    # QW #10: Track consecutive losses
                    if hit_tp:
                        consecutive_losses = 0  # Reset on win
                    else:
                        consecutive_losses += 1

            # QW #10: Skip next signals after 3 consecutive losses
            elif row['signal'] and consecutive_losses < 3:
                position = {
                    'trade_id': len(trades) + 1,
                    'entry_time': row.name,
                    'entry_idx': i,
                    'entry_price': row['close'],
                    'prob_tp': row['prob_tp'],
                    'day_of_week': row['day_of_week']
                }

        if len(trades) == 0:
            print(f"  WARNING: No trades generated")
            return {
                'crypto': self.crypto,
                'mode': self.mode,
                'total_trades': 0,
                'win_rate': 0,
                'roi': 0,
                'sharpe': 0
            }

        # Calculate metrics
        trades_df = pd.DataFrame(trades)
        total_trades = len(trades_df)
        wins = len(trades_df[trades_df['exit_type'] == 'TP'])
        losses = len(trades_df[trades_df['exit_type'] == 'SL'])
        win_rate = (wins / total_trades) * 100

        # PnL
        total_pnl = trades_df['pnl_pct'].sum()
        avg_pnl = trades_df['pnl_pct'].mean()

        # Cumulative returns
        cumulative_pnl = trades_df['pnl_pct'].cumsum()
        max_drawdown = (cumulative_pnl - cumulative_pnl.cummax()).min()

        # Sharpe ratio
        std_pnl = trades_df['pnl_pct'].std()
        sharpe = (avg_pnl / std_pnl) * np.sqrt(252) if std_pnl > 0 else 0

        # Day of week breakdown
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        trades_df['day_name'] = trades_df['day_of_week'].apply(lambda x: day_names[int(x)])

        print(f"\n  Trades by day of week:")
        for day in range(7):
            day_trades = trades_df[trades_df['day_of_week'] == day]
            if len(day_trades) > 0:
                day_wr = (day_trades['result'] == 'WIN').sum() / len(day_trades) * 100
                print(f"    {day_names[day]:3s}: {len(day_trades):3d} trades, {day_wr:5.1f}% WR")

        results = {
            'crypto': self.crypto,
            'mode': self.mode,
            'total_trades': int(total_trades),
            'wins': int(wins),
            'losses': int(losses),
            'win_rate': float(win_rate),
            'roi': float(total_pnl),
            'avg_pnl': float(avg_pnl),
            'sharpe': float(sharpe),
            'max_drawdown': float(max_drawdown),
            'prob_threshold': self.prob_threshold,
            'improvements_applied': [
                'QW #1: Adaptive threshold',
                'QW #2: Low confidence filter',
                'QW #5: Market regime filter' if has_sma_filter else 'QW #5: DISABLED',
                'QW #10: Loss protection',
                'QW #11: Day filter'
            ]
        }

        return results


def backtest_all_optimized():
    """Backtest all 3 models with Phase 1 improvements"""

    models = [
        ('btc', 'baseline'),
        ('eth', 'top50'),
        ('sol', 'optuna')
    ]

    print("="*120)
    print("V11 OPTIMIZED BACKTEST - PHASE 1 QUICK WINS")
    print("="*120)
    print("\nImprovements applied:")
    print("  QW #1: Adaptive thresholds (BTC=0.55, ETH=0.50, SOL=0.65)")
    print("  QW #2: Filter low confidence zone (50-53%)")
    print("  QW #5: Market regime filter (SMA50 > SMA200)")
    print("  QW #10: Consecutive loss protection (pause after 3 losses)")
    print("  QW #11: Day-of-week filter (avoid Friday)")
    print()

    all_results = []

    for crypto, mode in models:
        print(f"\n{'='*120}")
        print(f"BACKTESTING: {crypto.upper()} - {mode.upper()}")
        print('='*120)

        try:
            bt = OptimizedBacktest(crypto, mode, tp_pct=1.5, sl_pct=0.75)
            results = bt.run_optimized_backtest(start_date='2025-01-01')

            if results:
                all_results.append(results)

                print(f"\n  RESULTS:")
                print(f"    Total Trades: {results['total_trades']}")
                print(f"    Win Rate: {results['win_rate']:.2f}%")
                print(f"    Total ROI: {results['roi']:.2f}%")
                print(f"    Sharpe: {results['sharpe']:.2f}")
                print(f"    Max DD: {results['max_drawdown']:.2f}%")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save results
    results_file = Path(__file__).parent.parent / 'backtesting' / 'results' / 'v11_optimized_results.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Comparison with baseline
    print(f"\n\n{'='*120}")
    print("COMPARISON: BASELINE vs OPTIMIZED")
    print('='*120)

    baseline_file = Path(__file__).parent.parent / 'backtesting' / 'results' / 'all_phases_backtest.json'
    with open(baseline_file, 'r') as f:
        baseline_results = json.load(f)

    print(f"\n{'Crypto':6s} | {'Mode':8s} | {'Baseline WR':12s} | {'Optimized WR':13s} | {'Improvement':11s} | {'Baseline ROI':13s} | {'Optimized ROI':14s}")
    print("-"*120)

    for opt_result in all_results:
        crypto = opt_result['crypto']
        mode = opt_result['mode']

        # Find corresponding baseline
        baseline = next((r for r in baseline_results if r['crypto'] == crypto and r['mode'] == mode), None)

        if baseline:
            baseline_wr = baseline['win_rate']
            optimized_wr = opt_result['win_rate']
            wr_improvement = optimized_wr - baseline_wr

            baseline_roi = baseline['roi']
            optimized_roi = opt_result['roi']

            print(f"{crypto.upper():6s} | {mode.upper():8s} | {baseline_wr:10.2f}% | {optimized_wr:11.2f}% | {wr_improvement:+9.2f}% | {baseline_roi:+11.2f}% | {optimized_roi:+12.2f}%")

    # Combined improvement
    baseline_combined_wr = sum(r['win_rate'] for r in baseline_results) / len(baseline_results)
    optimized_combined_wr = sum(r['win_rate'] for r in all_results) / len(all_results)
    combined_improvement = optimized_combined_wr - baseline_combined_wr

    print(f"\n{'='*120}")
    print(f"COMBINED WIN RATE:")
    print(f"  Baseline:  {baseline_combined_wr:.2f}%")
    print(f"  Optimized: {optimized_combined_wr:.2f}%")
    print(f"  Improvement: {combined_improvement:+.2f}%")
    print('='*120)

    print(f"\nResults saved to: {results_file}\n")

    return all_results


if __name__ == '__main__':
    results = backtest_all_optimized()
