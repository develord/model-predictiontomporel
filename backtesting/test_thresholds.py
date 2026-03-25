"""
Test Multiple Probability Thresholds
=====================================
Compare Win Rate and ROI across different probability thresholds
"""

import sys
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent))


class ThresholdTester:
    """Test different probability thresholds"""

    def __init__(self, crypto: str, mode: str, tp_pct=1.5, sl_pct=0.75):
        self.crypto = crypto
        self.mode = mode
        self.tp_pct = tp_pct
        self.sl_pct = sl_pct

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

    def test_threshold(self, prob_threshold: float, start_date='2025-01-01'):
        """Test a specific threshold"""

        # Filter to test period
        df_test = self.df[self.df.index >= start_date].copy()

        # Prepare features
        X_test = df_test[self.feature_cols].fillna(0).values
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

        # Predict
        prob_tp = self.model.predict_proba(X_test)[:, 1]
        df_test['prob_tp'] = prob_tp
        df_test['signal'] = prob_tp > prob_threshold

        # Simulate trades
        trades = []
        position = None

        for i in range(len(df_test)):
            row = df_test.iloc[i]

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

                    position['exit_type'] = exit_type
                    position['pnl_pct'] = pnl_pct

                    trades.append(position)
                    position = None

            elif row['signal']:
                position = {
                    'entry_price': row['close'],
                    'prob_tp': row['prob_tp']
                }

        if len(trades) == 0:
            return {
                'threshold': prob_threshold,
                'total_trades': 0,
                'win_rate': 0,
                'roi': 0,
                'sharpe': 0
            }

        # Calculate metrics
        trades_df = pd.DataFrame(trades)
        total_trades = len(trades_df)
        wins = len(trades_df[trades_df['exit_type'] == 'TP'])
        win_rate = (wins / total_trades) * 100
        roi = trades_df['pnl_pct'].sum()

        avg_pnl = trades_df['pnl_pct'].mean()
        std_pnl = trades_df['pnl_pct'].std()
        sharpe = (avg_pnl / std_pnl) * np.sqrt(252) if std_pnl > 0 else 0

        return {
            'threshold': prob_threshold,
            'total_trades': int(total_trades),
            'wins': int(wins),
            'losses': int(total_trades - wins),
            'win_rate': float(win_rate),
            'roi': float(roi),
            'avg_pnl': float(avg_pnl),
            'sharpe': float(sharpe)
        }


def test_all_thresholds():
    """Test all models with multiple thresholds"""

    models = [
        ('btc', 'baseline'),
        ('eth', 'top50'),
        ('sol', 'optuna')
    ]

    thresholds = [0.50, 0.55, 0.60, 0.65]

    print("=" * 120)
    print("THRESHOLD OPTIMIZATION - PRODUCTION V11 MODELS")
    print("=" * 120)
    print()

    all_results = []

    for crypto, mode in models:
        print(f"\n{'='*120}")
        print(f"{crypto.upper()} {mode.upper()} - TESTING THRESHOLDS")
        print('='*120)
        print()

        try:
            tester = ThresholdTester(crypto, mode, tp_pct=1.5, sl_pct=0.75)

            crypto_results = []
            for threshold in thresholds:
                result = tester.test_threshold(threshold)
                result['crypto'] = crypto
                result['mode'] = mode
                crypto_results.append(result)
                all_results.append(result)

                print(f"  Threshold {threshold:.2f}: "
                      f"Trades={result['total_trades']:3d}, "
                      f"WR={result['win_rate']:5.2f}%, "
                      f"ROI={result['roi']:+7.2f}%, "
                      f"Sharpe={result['sharpe']:5.2f}")

            # Find best threshold for this crypto
            best_wr = max(crypto_results, key=lambda x: x['win_rate'])
            best_roi = max(crypto_results, key=lambda x: x['roi'])

            print()
            print(f"  Best Win Rate: Threshold={best_wr['threshold']:.2f} → WR={best_wr['win_rate']:.2f}%")
            print(f"  Best ROI:      Threshold={best_roi['threshold']:.2f} → ROI={best_roi['roi']:+.2f}%")

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    # Summary table
    print(f"\n\n{'='*120}")
    print("COMPREHENSIVE COMPARISON TABLE")
    print('='*120)
    print()

    print(f"{'Crypto':<8} {'Mode':<10} {'Threshold':<12} {'Trades':<8} {'Wins':<6} {'Losses':<8} "
          f"{'Win Rate':<10} {'ROI':<12} {'Sharpe':<8}")
    print("-" * 120)

    for r in all_results:
        print(f"{r['crypto'].upper():<8} {r['mode'].upper():<10} {r['threshold']:<12.2f} "
              f"{r['total_trades']:<8} {r['wins']:<6} {r['losses']:<8} "
              f"{r['win_rate']:<10.2f} {r['roi']:+<12.2f} {r['sharpe']:<8.2f}")

    # Best overall per crypto
    print(f"\n\n{'='*120}")
    print("OPTIMAL THRESHOLD PER CRYPTO (Best ROI)")
    print('='*120)
    print()

    for crypto, mode in models:
        crypto_results = [r for r in all_results if r['crypto'] == crypto]
        if crypto_results:
            best = max(crypto_results, key=lambda x: x['roi'])
            print(f"{crypto.upper():<8} {mode.upper():<10} - Threshold={best['threshold']:.2f}, "
                  f"WR={best['win_rate']:.2f}%, ROI={best['roi']:+.2f}%, "
                  f"Trades={best['total_trades']}, Sharpe={best['sharpe']:.2f}")

    # Save results
    results_file = Path(__file__).parent.parent / 'backtesting' / 'results' / 'threshold_optimization.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n\nResults saved to: {results_file}")
    print()

    return all_results


if __name__ == '__main__':
    results = test_all_thresholds()
