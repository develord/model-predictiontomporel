"""
Backtest ALL V11 Phase Models
==============================
Backtest all 9 models (3 cryptos × 3 modes) and generate comprehensive comparison
"""

import sys
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))


class PhaseBacktest:
    """Backtest V11 phase models"""

    def __init__(self, crypto: str, mode: str, tp_pct=1.5, sl_pct=0.75, prob_threshold=0.5):
        """
        Args:
            crypto: 'btc', 'eth', or 'sol'
            mode: 'baseline', 'optuna', or 'top50'
            tp_pct: Take profit percentage
            sl_pct: Stop loss percentage
            prob_threshold: P(TP) threshold for entry
        """
        self.crypto = crypto
        self.mode = mode
        self.tp_pct = tp_pct
        self.sl_pct = sl_pct
        self.prob_threshold = prob_threshold

        # Load model
        model_file = Path(__file__).parent.parent / 'models' / f'{crypto}_v11_{mode}.joblib'

        if not model_file.exists():
            raise FileNotFoundError(f"Model not found: {model_file}")

        self.model = joblib.load(model_file)

        # Load stats to get features used
        stats_file = Path(__file__).parent.parent / 'models' / f'{crypto}_v11_{mode}_stats.json'
        with open(stats_file, 'r') as f:
            stats = json.load(f)

        # Load data
        cache_file = Path(__file__).parent.parent / 'data' / 'cache' / f'{crypto}_multi_tf_merged.csv'
        self.df = pd.read_csv(cache_file, index_col=0, parse_dates=True)

        # Prepare features (same as training)
        exclude_cols = [
            'open', 'high', 'low', 'close', 'volume',
            'label_class', 'label_numeric', 'price_target_pct',
            'future_price', 'triple_barrier_label'
        ]

        all_features = [col for col in self.df.columns if col not in exclude_cols]

        # For TOP50, get the exact features used
        if mode == 'top50':
            top_features = [f['feature'] for f in stats['top_features'][:50]]
            self.feature_cols = [f for f in top_features if f in self.df.columns]
        else:
            self.feature_cols = all_features

        print(f"Loaded {crypto.upper()} {mode.upper()} model")
        print(f"  Features: {len(self.feature_cols)}")

    def run_backtest(self, start_date='2025-01-01'):
        """
        Run backtest on 2025+ data (out-of-sample test period)

        Args:
            start_date: Start date for backtest (default 2025-01-01)

        Returns:
            results: Dictionary with backtest metrics
        """
        # Filter to test period (2025+)
        df_test = self.df[self.df.index >= start_date].copy()

        if len(df_test) == 0:
            print(f"  WARNING: No test data for {start_date}+")
            return None

        print(f"  Test period: {df_test.index[0]} to {df_test.index[-1]} ({len(df_test)} candles)")

        # Prepare features
        X_test = df_test[self.feature_cols].fillna(0).values
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

        # Predict P(TP)
        prob_tp = self.model.predict_proba(X_test)[:, 1]

        # Add to dataframe
        df_test['prob_tp'] = prob_tp
        df_test['signal'] = prob_tp > self.prob_threshold

        # Simulate trades
        trades = []
        position = None

        for i in range(len(df_test)):
            row = df_test.iloc[i]

            # Skip if in position
            if position is not None:
                # Check if TP/SL hit
                high = df_test['high'].iloc[i]
                low = df_test['low'].iloc[i]

                entry_price = position['entry_price']
                tp_price = entry_price * (1 + self.tp_pct / 100)
                sl_price = entry_price * (1 - self.sl_pct / 100)

                hit_tp = high >= tp_price
                hit_sl = low <= sl_price

                if hit_tp or hit_sl:
                    # Close position
                    exit_price = tp_price if hit_tp else sl_price
                    exit_type = 'TP' if hit_tp else 'SL'
                    pnl_pct = ((exit_price - entry_price) / entry_price) * 100

                    position['exit_time'] = row.name
                    position['exit_price'] = exit_price
                    position['exit_type'] = exit_type
                    position['pnl_pct'] = pnl_pct
                    position['bars_held'] = i - position['entry_idx']

                    trades.append(position)
                    position = None

            # Open new position if signal and no position
            elif row['signal']:
                position = {
                    'entry_time': row.name,
                    'entry_idx': i,
                    'entry_price': row['close'],
                    'prob_tp': row['prob_tp']
                }

        # Calculate metrics
        if len(trades) == 0:
            print(f"  WARNING: No trades generated")
            return {
                'crypto': self.crypto,
                'mode': self.mode,
                'total_trades': 0,
                'win_rate': 0,
                'roi': 0,
                'avg_pnl_tp': 0,
                'avg_pnl_sl': 0,
                'sharpe': 0,
                'max_drawdown': 0
            }

        # Convert to DataFrame
        trades_df = pd.DataFrame(trades)

        # Metrics
        total_trades = len(trades_df)
        wins = len(trades_df[trades_df['exit_type'] == 'TP'])
        losses = len(trades_df[trades_df['exit_type'] == 'SL'])
        win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0

        # PnL
        total_pnl = trades_df['pnl_pct'].sum()
        avg_pnl_tp = trades_df[trades_df['exit_type'] == 'TP']['pnl_pct'].mean() if wins > 0 else 0
        avg_pnl_sl = trades_df[trades_df['exit_type'] == 'SL']['pnl_pct'].mean() if losses > 0 else 0
        avg_pnl = trades_df['pnl_pct'].mean()

        # Cumulative returns
        cumulative_pnl = trades_df['pnl_pct'].cumsum()
        max_drawdown = (cumulative_pnl - cumulative_pnl.cummax()).min()

        # Sharpe ratio (annualized)
        std_pnl = trades_df['pnl_pct'].std()
        sharpe = (avg_pnl / std_pnl) * np.sqrt(365 / trades_df['bars_held'].mean()) if std_pnl > 0 else 0

        results = {
            'crypto': self.crypto,
            'mode': self.mode,
            'total_trades': int(total_trades),
            'wins': int(wins),
            'losses': int(losses),
            'win_rate': float(win_rate),
            'roi': float(total_pnl),
            'avg_pnl': float(avg_pnl),
            'avg_pnl_tp': float(avg_pnl_tp),
            'avg_pnl_sl': float(avg_pnl_sl),
            'sharpe': float(sharpe),
            'max_drawdown': float(max_drawdown),
            'avg_bars_held': float(trades_df['bars_held'].mean()),
            'prob_threshold': self.prob_threshold,
            'tp_pct': self.tp_pct,
            'sl_pct': self.sl_pct
        }

        return results


def backtest_all_models():
    """Backtest all 9 models and generate comparison"""

    cryptos = ['btc', 'eth', 'sol']
    modes = ['baseline', 'optuna', 'top50']

    print("=" * 100)
    print("BACKTEST ALL V11 PHASE MODELS - 2025+ OUT-OF-SAMPLE PERIOD")
    print("=" * 100)
    print()

    all_results = []

    for crypto in cryptos:
        for mode in modes:
            print(f"\n{'='*100}")
            print(f"BACKTESTING: {crypto.upper()} - {mode.upper()}")
            print('='*100)

            try:
                bt = PhaseBacktest(crypto, mode, tp_pct=1.5, sl_pct=0.75, prob_threshold=0.5)
                results = bt.run_backtest(start_date='2025-01-01')

                if results:
                    all_results.append(results)

                    print(f"\n  Results:")
                    print(f"    Total Trades: {results['total_trades']}")
                    print(f"    Win Rate: {results['win_rate']:.2f}%")
                    print(f"    Total ROI: {results['roi']:.2f}%")
                    print(f"    Avg PnL: {results['avg_pnl']:.2f}%")
                    print(f"    Avg PnL (TP): {results['avg_pnl_tp']:.2f}%")
                    print(f"    Avg PnL (SL): {results['avg_pnl_sl']:.2f}%")
                    print(f"    Sharpe: {results['sharpe']:.2f}")
                    print(f"    Max DD: {results['max_drawdown']:.2f}%")

            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
                continue

    # Save results
    results_file = Path(__file__).parent.parent / 'backtesting' / 'results' / 'all_phases_backtest.json'
    results_file.parent.mkdir(parents=True, exist_ok=True)

    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n\n{'='*100}")
    print("RESULTS SAVED TO: backtesting/results/all_phases_backtest.json")
    print('='*100)

    # Generate comparison table
    print(f"\n\n{'='*100}")
    print("BACKTEST SUMMARY - SORTED BY ROI")
    print('='*100)
    print()

    # Sort by ROI
    sorted_results = sorted(all_results, key=lambda x: x['roi'], reverse=True)

    print(f"{'Crypto':<8} {'Mode':<10} {'Trades':<8} {'Win Rate':<10} {'ROI':<10} {'Avg PnL':<10} {'Sharpe':<8} {'Max DD':<10}")
    print("-" * 100)

    for r in sorted_results:
        print(f"{r['crypto'].upper():<8} {r['mode'].upper():<10} {r['total_trades']:<8} "
              f"{r['win_rate']:.2f}%{'':<5} {r['roi']:+.2f}%{'':<4} {r['avg_pnl']:+.2f}%{'':<4} "
              f"{r['sharpe']:.2f}{'':<3} {r['max_drawdown']:.2f}%")

    # Best per crypto
    print(f"\n\n{'='*100}")
    print("BEST CONFIGURATION PER CRYPTO (by ROI)")
    print('='*100)
    print()

    for crypto in cryptos:
        crypto_results = [r for r in all_results if r['crypto'] == crypto]
        if crypto_results:
            best = max(crypto_results, key=lambda x: x['roi'])
            print(f"{crypto.upper()}: {best['mode'].upper()} - ROI: {best['roi']:+.2f}%, Win Rate: {best['win_rate']:.2f}%, "
                  f"Trades: {best['total_trades']}, Sharpe: {best['sharpe']:.2f}")

    print()

    return all_results


if __name__ == '__main__':
    results = backtest_all_models()
