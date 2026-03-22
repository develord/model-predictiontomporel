"""
V11 Backtest - Test period: 2026-01-01 to today
Detailed trade-by-trade results for BTC, ETH, SOL
"""

import sys
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent))


class V11BacktestEngine:
    """Backtest V11 models on 2026+ period"""

    def __init__(self, crypto: str, initial_capital: float = 10000.0):
        self.crypto = crypto.lower()
        self.initial_capital = initial_capital
        self.capital = initial_capital

        # Load model
        model_path = Path(__file__).parent.parent / 'models' / f'{self.crypto}_v11_classifier.joblib'
        self.model = joblib.load(model_path)

        # Trading params (from V11 config)
        self.tp_threshold = 0.60  # 60% confidence required
        self.fixed_tp_pct = 1.5   # +1.5% take profit
        self.fixed_sl_pct = 0.75  # -0.75% stop loss
        self.position_size_pct = 10.0  # 10% of capital per trade
        self.fee_pct = 0.1  # 0.1% fee per trade

        # Results
        self.trades = []
        self.equity_curve = []

    def load_test_data(self):
        """Load test data from 2026-01-01 onwards"""
        cache_file = Path(__file__).parent.parent / 'data' / 'cache' / f'{self.crypto}_multi_tf_merged.csv'

        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)

        # Filter for test period (2026+)
        test_df = df[df.index >= '2026-01-01'].copy()

        print(f"\n{self.crypto.upper()} Test Data:")
        print(f"  Period: {test_df.index[0]} to {test_df.index[-1]}")
        print(f"  Candles: {len(test_df)}")

        return test_df

    def prepare_features(self, df):
        """Extract features for prediction"""
        exclude_cols = [
            'open', 'high', 'low', 'close', 'volume',
            'label_class', 'label_numeric',
            'price_target_pct', 'future_price',
            'triple_barrier_label'
        ]

        feature_cols = [col for col in df.columns if col not in exclude_cols]
        return feature_cols

    def simulate_trade(self, entry_idx, entry_price, tp_price, sl_price, prob_tp, future_df):
        """Simulate a single trade until TP or SL hit"""

        position_value = self.capital * (self.position_size_pct / 100)

        # Scan future candles
        for i in range(min(len(future_df), 100)):
            candle = future_df.iloc[i]
            high = candle['high']
            low = candle['low']

            # Check TP first (priority)
            if high >= tp_price:
                pnl_pct = ((tp_price - entry_price) / entry_price) * 100
                pnl_pct -= self.fee_pct * 2  # Entry + exit fees
                pnl = position_value * (pnl_pct / 100)

                return {
                    'entry_time': future_df.index[0],
                    'exit_time': candle.name,
                    'entry_price': entry_price,
                    'exit_price': tp_price,
                    'result': 'WIN',
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'capital_before': self.capital,
                    'capital_after': self.capital + pnl,
                    'prob_tp': prob_tp,
                    'hold_periods': i + 1
                }

            # Check SL
            if low <= sl_price:
                pnl_pct = ((sl_price - entry_price) / entry_price) * 100
                pnl_pct -= self.fee_pct * 2  # Entry + exit fees
                pnl = position_value * (pnl_pct / 100)

                return {
                    'entry_time': future_df.index[0],
                    'exit_time': candle.name,
                    'entry_price': entry_price,
                    'exit_price': sl_price,
                    'result': 'LOSE',
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'capital_before': self.capital,
                    'capital_after': self.capital + pnl,
                    'prob_tp': prob_tp,
                    'hold_periods': i + 1
                }

        return None  # Timeout (no TP/SL hit)

    def run_backtest(self):
        """Run backtest on 2026+ data"""

        print(f"\n{'='*80}")
        print(f"V11 BACKTEST - {self.crypto.upper()}")
        print(f"{'='*80}")
        print(f"Threshold: P(TP) > {self.tp_threshold}")
        print(f"TP: +{self.fixed_tp_pct}%, SL: -{self.fixed_sl_pct}%")
        print(f"Position size: {self.position_size_pct}% of capital")
        print(f"Initial capital: ${self.initial_capital:,.2f}")

        # Load data
        test_df = self.load_test_data()
        feature_cols = self.prepare_features(test_df)

        # Iterate through test period
        for i in range(len(test_df) - 1):
            idx = test_df.index[i]
            row = test_df.loc[idx]

            # Extract features
            features = row[feature_cols].fillna(0).values.reshape(1, -1)
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

            # Predict P(TP)
            prob_tp = self.model.predict_proba(features)[0, 1]

            # Trading decision
            if prob_tp < self.tp_threshold:
                continue  # Skip - not confident enough

            # Open trade
            entry_price = row['close']
            tp_price = entry_price * (1 + self.fixed_tp_pct / 100)
            sl_price = entry_price * (1 - self.fixed_sl_pct / 100)

            # Simulate trade
            trade_result = self.simulate_trade(
                i,
                entry_price,
                tp_price,
                sl_price,
                prob_tp,
                test_df.iloc[i:]
            )

            if trade_result:
                self.trades.append(trade_result)
                self.capital = trade_result['capital_after']

                self.equity_curve.append({
                    'timestamp': idx,
                    'capital': self.capital,
                    'roi': (self.capital / self.initial_capital - 1) * 100
                })

        return self.calculate_metrics()

    def calculate_metrics(self):
        """Calculate performance metrics"""

        if len(self.trades) == 0:
            return {
                'error': 'No trades executed',
                'crypto': self.crypto
            }

        trades_df = pd.DataFrame(self.trades)

        # Basic metrics
        total_trades = len(trades_df)
        wins = len(trades_df[trades_df['result'] == 'WIN'])
        losses = len(trades_df[trades_df['result'] == 'LOSE'])
        win_rate = wins / total_trades * 100 if total_trades > 0 else 0

        # ROI
        total_roi = (self.capital / self.initial_capital - 1) * 100

        # PnL
        total_pnl = trades_df['pnl'].sum()
        avg_win = trades_df[trades_df['result'] == 'WIN']['pnl'].mean() if wins > 0 else 0
        avg_loss = trades_df[trades_df['result'] == 'LOSE']['pnl'].mean() if losses > 0 else 0

        # Profit factor
        total_wins_pnl = trades_df[trades_df['result'] == 'WIN']['pnl'].sum()
        total_losses_pnl = abs(trades_df[trades_df['result'] == 'LOSE']['pnl'].sum())
        profit_factor = total_wins_pnl / total_losses_pnl if total_losses_pnl > 0 else 0

        # Max drawdown
        if len(self.equity_curve) > 0:
            equity_df = pd.DataFrame(self.equity_curve)
            equity_df['peak'] = equity_df['capital'].cummax()
            equity_df['drawdown'] = (equity_df['capital'] - equity_df['peak']) / equity_df['peak'] * 100
            max_drawdown = equity_df['drawdown'].min()
        else:
            max_drawdown = 0

        metrics = {
            'crypto': self.crypto.upper(),
            'initial_capital': float(self.initial_capital),
            'final_capital': float(self.capital),
            'total_roi': float(total_roi),
            'total_pnl': float(total_pnl),
            'total_trades': int(total_trades),
            'wins': int(wins),
            'losses': int(losses),
            'win_rate': float(win_rate),
            'avg_win': float(avg_win),
            'avg_loss': float(avg_loss),
            'profit_factor': float(profit_factor),
            'max_drawdown': float(max_drawdown),
            'avg_hold_periods': float(trades_df['hold_periods'].mean()),
            'avg_prob_tp': float(trades_df['prob_tp'].mean())
        }

        return metrics, trades_df


def backtest_all_cryptos():
    """Run backtest for all cryptos"""

    print("\n" + "="*80)
    print("V11 BACKTEST - 2026 TEST PERIOD")
    print("="*80)
    print(f"Test period: 2026-01-01 to today ({datetime.now().strftime('%Y-%m-%d')})")
    print("="*80)

    cryptos = ['btc', 'eth', 'sol']
    all_results = {}
    all_trades = {}

    for crypto in cryptos:
        try:
            engine = V11BacktestEngine(crypto, initial_capital=10000.0)
            metrics, trades_df = engine.run_backtest()

            all_results[crypto] = metrics
            all_trades[crypto] = trades_df

        except Exception as e:
            print(f"\nERROR backtesting {crypto}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Print summary
    print("\n\n" + "="*80)
    print("BACKTEST SUMMARY - ALL CRYPTOS")
    print("="*80)

    for crypto in cryptos:
        if crypto not in all_results:
            continue

        metrics = all_results[crypto]

        if 'error' in metrics:
            print(f"\n{crypto.upper()}: {metrics['error']}")
            continue

        print(f"\n{crypto.upper()}:")
        print(f"  Total Trades: {metrics['total_trades']}")
        print(f"  Wins: {metrics['wins']} | Losses: {metrics['losses']}")
        print(f"  Win Rate: {metrics['win_rate']:.2f}%")
        print(f"  Total ROI: {metrics['total_roi']:+.2f}%")
        print(f"  Total P&L: ${metrics['total_pnl']:+,.2f}")
        print(f"  Final Capital: ${metrics['final_capital']:,.2f}")
        print(f"  Avg Win: ${metrics['avg_win']:+.2f}")
        print(f"  Avg Loss: ${metrics['avg_loss']:+.2f}")
        print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"  Max Drawdown: {metrics['max_drawdown']:.2f}%")

    # Print detailed trades
    print("\n\n" + "="*80)
    print("DETAILED TRADE LIST")
    print("="*80)

    for crypto in cryptos:
        if crypto not in all_trades or all_trades[crypto] is None:
            continue

        trades_df = all_trades[crypto]

        if len(trades_df) == 0:
            print(f"\n{crypto.upper()}: No trades")
            continue

        print(f"\n{crypto.upper()} - {len(trades_df)} trades:")
        print("-" * 80)

        for idx, trade in trades_df.iterrows():
            result_text = "[WIN] " if trade['result'] == 'WIN' else "[LOSE]"

            print(f"{idx+1:2d}. {result_text} | "
                  f"Entry: {trade['entry_time'].strftime('%Y-%m-%d %H:%M')} | "
                  f"Exit: {trade['exit_time'].strftime('%Y-%m-%d %H:%M')} | "
                  f"P&L: ${trade['pnl']:+7.2f} ({trade['pnl_pct']:+.2f}%) | "
                  f"Capital: ${trade['capital_after']:,.2f} | "
                  f"Confidence: {trade['prob_tp']:.1%}")

        print(f"\nTotal: {trades_df['pnl'].sum():+.2f} USD")

    # Save results
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics
    with open(results_dir / 'backtest_2026_metrics.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # Save trades
    for crypto, trades_df in all_trades.items():
        if trades_df is not None and len(trades_df) > 0:
            trades_df.to_csv(results_dir / f'backtest_2026_{crypto}_trades.csv')

    print("\n" + "="*80)
    print("BACKTEST COMPLETE!")
    print(f"Results saved to: {results_dir}")
    print("="*80)

    return all_results, all_trades


if __name__ == '__main__':
    backtest_all_cryptos()
