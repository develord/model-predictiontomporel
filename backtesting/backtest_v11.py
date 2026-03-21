"""
V11 PRO Backtest - Binary Classifier TP/SL Trading
===================================================
Backtest V11 optimized models with fixed TP/SL thresholds

Trading Strategy:
- Load optimized V11 binary classifier
- Predict P(TP) for each candle
- If P(TP) > threshold: Open trade with TP=1.5%, SL=0.75%
- Hold position until TP or SL hit
- Track all metrics (ROI, Sharpe, win rate, etc.)

Comparison:
- V10 baseline (failed with ~30% accuracy)
- V11 baseline (~52% accuracy)
- V11 optimized (~55-63% accuracy)
"""

import sys
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
import json

sys.path.insert(0, str(Path(__file__).parent.parent))


class V11Backtest:
    """Backtest V11 binary classifier with fixed TP/SL"""

    def __init__(self, crypto: str, model_type='optimized', tp_pct=1.5, sl_pct=0.75, prob_threshold=0.5):
        """
        Args:
            crypto: 'btc', 'eth', or 'sol'
            model_type: 'baseline' or 'optimized'
            tp_pct: Take profit percentage (default 1.5%)
            sl_pct: Stop loss percentage (default 0.75%)
            prob_threshold: P(TP) threshold for entry (default 0.5)
        """
        self.crypto = crypto
        self.model_type = model_type
        self.tp_pct = tp_pct
        self.sl_pct = sl_pct
        self.prob_threshold = prob_threshold

        # Load model
        if model_type == 'optimized':
            model_file = Path(__file__).parent.parent / 'models' / f'{crypto}_v11_optimized.joblib'
        else:
            model_file = Path(__file__).parent.parent / 'models' / f'{crypto}_v11_classifier.joblib'

        if not model_file.exists():
            raise FileNotFoundError(f"Model not found: {model_file}")

        self.model = joblib.load(model_file)

        # Load data
        cache_file = Path(__file__).parent.parent / 'data' / 'cache' / f'{crypto}_multi_tf_merged.csv'
        self.df = pd.read_csv(cache_file, index_col=0, parse_dates=True)

        # Prepare features
        exclude_cols = [
            'open', 'high', 'low', 'close', 'volume',
            'label_class', 'label_numeric', 'price_target_pct',
            'future_price', 'triple_barrier_label'
        ]
        self.feature_cols = [col for col in self.df.columns if col not in exclude_cols]

        print(f"Loaded {crypto.upper()} {model_type} model")
        print(f"  Data: {len(self.df)} candles")
        print(f"  Features: {len(self.feature_cols)}")
        print(f"  TP/SL: {self.tp_pct}% / {self.sl_pct}%")
        print(f"  P(TP) threshold: {self.prob_threshold}")


    def run_backtest(self, test_ratio=0.2):
        """
        Run backtest on test data

        Args:
            test_ratio: Fraction of data for testing (default 0.2 = last 20%)

        Returns:
            results: Dictionary with backtest metrics
        """
        # Test split
        split_idx = int(len(self.df) * (1 - test_ratio))
        df_test = self.df.iloc[split_idx:].copy()

        print(f"\nRunning backtest on test data ({len(df_test)} candles)...")

        # Prepare features
        X_test = df_test[self.feature_cols].fillna(0).values
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

        # Predict P(TP)
        prob_tp = self.model.predict_proba(X_test)[:, 1]

        # Add to dataframe
        df_test['prob_tp'] = prob_tp
        df_test['signal'] = (prob_tp > self.prob_threshold).astype(int)

        # Simulate trades
        trades = self._simulate_trades(df_test)

        # Calculate metrics
        results = self._calculate_metrics(trades, df_test)

        return results, trades, df_test


    def _simulate_trades(self, df):
        """Simulate trades with fixed TP/SL"""
        trades = []
        in_position = False
        entry_idx = None
        entry_price = None

        for idx in range(len(df)):
            row = df.iloc[idx]

            # Check if we should enter
            if not in_position and row['signal'] == 1:
                in_position = True
                entry_idx = idx
                entry_price = row['close']
                continue

            # Check if we're in position
            if in_position:
                current_price = row['close']

                # Calculate P&L %
                pnl_pct = ((current_price - entry_price) / entry_price) * 100

                # Check TP
                if pnl_pct >= self.tp_pct:
                    trades.append({
                        'entry_idx': entry_idx,
                        'exit_idx': idx,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'pnl_pct': self.tp_pct,  # Assume exact TP
                        'outcome': 'TP',
                        'bars_held': idx - entry_idx
                    })
                    in_position = False
                    continue

                # Check SL
                if pnl_pct <= -self.sl_pct:
                    trades.append({
                        'entry_idx': entry_idx,
                        'exit_idx': idx,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'pnl_pct': -self.sl_pct,  # Assume exact SL
                        'outcome': 'SL',
                        'bars_held': idx - entry_idx
                    })
                    in_position = False
                    continue

        # Close any open position at end
        if in_position:
            final_price = df.iloc[-1]['close']
            final_pnl = ((final_price - entry_price) / entry_price) * 100
            trades.append({
                'entry_idx': entry_idx,
                'exit_idx': len(df) - 1,
                'entry_price': entry_price,
                'exit_price': final_price,
                'pnl_pct': final_pnl,
                'outcome': 'OPEN',
                'bars_held': len(df) - 1 - entry_idx
            })

        return trades


    def _calculate_metrics(self, trades, df):
        """Calculate backtest performance metrics"""
        if len(trades) == 0:
            return {
                'total_trades': 0,
                'roi': 0,
                'sharpe': 0,
                'win_rate': 0,
                'message': 'No trades executed'
            }

        trades_df = pd.DataFrame(trades)

        # Basic stats
        total_trades = len(trades_df)
        tp_trades = len(trades_df[trades_df['outcome'] == 'TP'])
        sl_trades = len(trades_df[trades_df['outcome'] == 'SL'])
        open_trades = len(trades_df[trades_df['outcome'] == 'OPEN'])

        win_rate = tp_trades / total_trades if total_trades > 0 else 0

        # ROI
        total_roi = trades_df['pnl_pct'].sum()
        avg_trade_roi = trades_df['pnl_pct'].mean()

        # Sharpe ratio (annualized)
        returns = trades_df['pnl_pct'].values
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(365 / 4) if np.std(returns) > 0 else 0

        # Average holding time
        avg_bars_held = trades_df['bars_held'].mean()

        # Max drawdown (simplified)
        cumulative_roi = np.cumsum(trades_df['pnl_pct'].values)
        running_max = np.maximum.accumulate(cumulative_roi)
        drawdown = running_max - cumulative_roi
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0

        # Expected value per trade
        ev = (self.tp_pct * win_rate) + (-self.sl_pct * (1 - win_rate))

        results = {
            'crypto': self.crypto,
            'model_type': self.model_type,
            'total_trades': total_trades,
            'tp_trades': tp_trades,
            'sl_trades': sl_trades,
            'open_trades': open_trades,
            'win_rate': win_rate,
            'total_roi': total_roi,
            'avg_trade_roi': avg_trade_roi,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'avg_bars_held': avg_bars_held,
            'expected_value': ev,
            'tp_pct': self.tp_pct,
            'sl_pct': self.sl_pct,
            'prob_threshold': self.prob_threshold
        }

        return results


    def print_results(self, results):
        """Print backtest results"""
        print(f"\n{'='*80}")
        print(f"BACKTEST RESULTS: {self.crypto.upper()} - {self.model_type.upper()}")
        print('='*80)

        if results['total_trades'] == 0:
            print(f"\nNo trades executed (P(TP) threshold too high: {self.prob_threshold})")
            return

        print(f"\nTRADE STATISTICS:")
        print(f"  Total trades: {results['total_trades']}")
        print(f"  TP trades: {results['tp_trades']} ({results['tp_trades']/results['total_trades']*100:.1f}%)")
        print(f"  SL trades: {results['sl_trades']} ({results['sl_trades']/results['total_trades']*100:.1f}%)")
        print(f"  Open trades: {results['open_trades']}")
        print(f"  Win rate: {results['win_rate']*100:.2f}%")

        print(f"\nPERFORMANCE:")
        print(f"  Total ROI: {results['total_roi']:.2f}%")
        print(f"  Avg trade ROI: {results['avg_trade_roi']:.2f}%")
        print(f"  Expected value/trade: {results['expected_value']:.2f}%")
        print(f"  Sharpe ratio: {results['sharpe_ratio']:.3f}")
        print(f"  Max drawdown: {results['max_drawdown']:.2f}%")

        print(f"\nTRADING PARAMETERS:")
        print(f"  TP: {results['tp_pct']}%")
        print(f"  SL: {results['sl_pct']}%")
        print(f"  P(TP) threshold: {results['prob_threshold']}")
        print(f"  Avg bars held: {results['avg_bars_held']:.1f}")

        # Verdict
        print(f"\nVERDICT:")
        if results['expected_value'] > 0.1:
            print(f"  PROFITABLE STRATEGY (EV = {results['expected_value']:.2f}% > 0)")
        elif results['expected_value'] > 0:
            print(f"  MARGINALLY PROFITABLE (EV = {results['expected_value']:.2f}%)")
        else:
            print(f"  UNPROFITABLE (EV = {results['expected_value']:.2f}% < 0)")


def backtest_all_models(cryptos=['btc', 'eth', 'sol']):
    """Backtest all cryptos with both baseline and optimized models"""

    print("="*80)
    print("V11 COMPREHENSIVE BACKTEST")
    print("="*80)
    print("\nTesting both baseline and optimized models on test data")
    print(f"Cryptos: {', '.join([c.upper() for c in cryptos])}")

    all_results = {}

    for crypto in cryptos:
        for model_type in ['baseline', 'optimized']:
            try:
                print(f"\n{'='*80}")
                print(f"TESTING: {crypto.upper()} - {model_type.upper()}")
                print('='*80)

                backtest = V11Backtest(crypto, model_type=model_type)
                results, trades, df_test = backtest.run_backtest()
                backtest.print_results(results)

                all_results[f'{crypto}_{model_type}'] = results

            except Exception as e:
                print(f"\nERROR backtesting {crypto} {model_type}: {e}")
                import traceback
                traceback.print_exc()
                continue

    # Summary comparison
    print(f"\n\n{'='*80}")
    print("SUMMARY: BASELINE VS OPTIMIZED")
    print('='*80)

    for crypto in cryptos:
        baseline_key = f'{crypto}_baseline'
        optimized_key = f'{crypto}_optimized'

        if baseline_key not in all_results or optimized_key not in all_results:
            print(f"\n{crypto.upper()}: INCOMPLETE DATA")
            continue

        baseline = all_results[baseline_key]
        optimized = all_results[optimized_key]

        print(f"\n{crypto.upper()}:")
        print(f"  Baseline:  WR={baseline['win_rate']*100:.1f}%, EV={baseline['expected_value']:.2f}%, ROI={baseline['total_roi']:.1f}%")
        print(f"  Optimized: WR={optimized['win_rate']*100:.1f}%, EV={optimized['expected_value']:.2f}%, ROI={optimized['total_roi']:.1f}%")

        ev_improvement = optimized['expected_value'] - baseline['expected_value']
        roi_improvement = optimized['total_roi'] - baseline['total_roi']

        print(f"  Improvement: EV {ev_improvement:+.2f}%, ROI {roi_improvement:+.1f}%")

        if optimized['expected_value'] > baseline['expected_value']:
            print(f"  OPTIMIZED IS BETTER!")
        else:
            print(f"  Baseline performed better")

    # Save results
    results_dir = Path(__file__).parent.parent / 'backtesting' / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = results_dir / f'v11_backtest_{timestamp}.json'

    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n\nResults saved to: {results_file}")

    print(f"\n{'='*80}")
    print("V11 BACKTEST COMPLETE!")
    print('='*80)

    return all_results


if __name__ == '__main__':
    results = backtest_all_models()
