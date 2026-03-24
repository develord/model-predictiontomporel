"""
V12 Backtest - V11 Model + Dynamic ATR TP/SL at Execution
==========================================================
Model predicts P(TP) same as V11, but TP/SL adapt to current ATR.
- High volatility: wider TP/SL (fewer premature stops)
- Low volatility: tighter TP/SL (faster trades)

Compare side-by-side with V11 fixed TP/SL (1.5%/0.75%).
"""

import sys
import pandas as pd
import numpy as np
import json
from pathlib import Path
import joblib

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from v12.features.dynamic_labels import calculate_atr_series


class V12BacktestEngine:
    """Backtest with dynamic ATR-based TP/SL at execution"""

    def __init__(self, crypto: str, initial_capital: float = 10000.0):
        self.crypto = crypto.lower()
        self.initial_capital = initial_capital
        self.capital = initial_capital

        # Load V12 model
        v12_models = Path(__file__).parent.parent / 'models'
        self.model = joblib.load(v12_models / f'{self.crypto}_v12_dynamic_atr.joblib')

        # Load feature list (ensures exact match with training)
        with open(v12_models / f'{self.crypto}_v12_features.json', 'r') as f:
            self.feature_cols = json.load(f)['feature_cols']

        # Load config
        config_path = Path(__file__).parent.parent / 'config' / 'v12_config.json'
        with open(config_path, 'r') as f:
            config = json.load(f)

        crypto_cfg = config.get(crypto.upper(), {})
        global_cfg = config.get('dynamic_tp_sl', {})
        trading_cfg = config.get('trading', {})

        self.tp_atr_mult = crypto_cfg.get('tp_atr_multiplier', global_cfg.get('tp_atr_multiplier', 0.40))
        self.sl_atr_mult = crypto_cfg.get('sl_atr_multiplier', global_cfg.get('sl_atr_multiplier', 0.15))
        self.atr_period = global_cfg.get('atr_period', 14)
        self.min_tp = global_cfg.get('min_tp_pct', 0.75)
        self.max_tp = global_cfg.get('max_tp_pct', 3.5)
        self.min_sl = global_cfg.get('min_sl_pct', 0.35)
        self.max_sl = global_cfg.get('max_sl_pct', 1.75)
        self.min_rr = global_cfg.get('risk_reward_min', 1.8)

        self.confidence_threshold = crypto_cfg.get('confidence_threshold', 0.35)
        self.position_size_pct = trading_cfg.get('position_size_pct', 10.0)
        self.fee_pct = trading_cfg.get('fee_pct', 0.1)

        self.trades = []
        self.equity_curve = []

    def load_test_data(self):
        """Load Q1 2025 test data with ATR + LSTM features"""
        cache_file = PROJECT_ROOT / 'data' / 'cache' / f'{self.crypto}_multi_tf_merged.csv'
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)

        # Add ATR as feature
        df['atr_pct_14'] = calculate_atr_series(df, self.atr_period)

        # Add LSTM features if model needs them
        lstm_cols = ['lstm_proba', 'lstm_confidence', 'lstm_signal', 'lstm_agrees_rsi']
        needs_lstm = any(c in self.feature_cols for c in lstm_cols)

        if needs_lstm:
            try:
                from v12.features.lstm_features import build_lstm_features_for_crypto
                lstm_features, _ = build_lstm_features_for_crypto(
                    self.crypto, lstm_train_end='2024-01-01',
                    seq_len=20, epochs=30, verbose=False
                )
                for col in lstm_cols:
                    df[col] = lstm_features[col].values
                print(f"  LSTM features generated")
            except Exception as e:
                print(f"  LSTM features failed ({e}), using defaults")
                for col in lstm_cols:
                    df[col] = 0.5 if 'proba' in col or 'confidence' in col else 0

        # Filter Q1 2025
        test_df = df[(df.index >= '2025-01-01') & (df.index <= '2025-03-31')].copy()

        print(f"\n  {self.crypto.upper()} Q1 2025: {len(test_df)} candles")
        print(f"  ATR mean: {test_df['atr_pct_14'].mean():.2f}%")

        return test_df

    def dynamic_tp_sl(self, atr_pct: float):
        """Compute TP/SL from current ATR"""
        tp = np.clip(atr_pct * self.tp_atr_mult, self.min_tp, self.max_tp)
        sl = np.clip(atr_pct * self.sl_atr_mult, self.min_sl, self.max_sl)
        if tp / sl < self.min_rr:
            sl = tp / self.min_rr
            sl = max(sl, self.min_sl)
        return tp, sl

    def simulate_trade(self, entry_price, tp_pct, sl_pct, prob_tp, future_df):
        """Simulate trade until TP or SL hit"""
        pos_value = self.capital * (self.position_size_pct / 100)
        tp_price = entry_price * (1 + tp_pct / 100)
        sl_price = entry_price * (1 - sl_pct / 100)

        for i in range(min(len(future_df), 100)):
            h = future_df.iloc[i]['high']
            l = future_df.iloc[i]['low']

            if h >= tp_price:
                pnl_pct = tp_pct - self.fee_pct * 2
                pnl = pos_value * pnl_pct / 100
                return {
                    'entry_time': future_df.index[0], 'exit_time': future_df.index[i],
                    'entry_price': entry_price, 'exit_price': tp_price,
                    'result': 'WIN', 'pnl': pnl, 'pnl_pct': pnl_pct,
                    'capital_after': self.capital + pnl, 'prob_tp': prob_tp,
                    'tp_pct': tp_pct, 'sl_pct': sl_pct, 'hold': i + 1
                }

            if l <= sl_price:
                pnl_pct = -sl_pct - self.fee_pct * 2
                pnl = pos_value * pnl_pct / 100
                return {
                    'entry_time': future_df.index[0], 'exit_time': future_df.index[i],
                    'entry_price': entry_price, 'exit_price': sl_price,
                    'result': 'LOSE', 'pnl': pnl, 'pnl_pct': pnl_pct,
                    'capital_after': self.capital + pnl, 'prob_tp': prob_tp,
                    'tp_pct': tp_pct, 'sl_pct': sl_pct, 'hold': i + 1
                }
        return None

    def run_backtest(self):
        print(f"\n{'='*80}")
        print(f"V12 BACKTEST - {self.crypto.upper()} (Dynamic ATR Execution)")
        print(f"{'='*80}")
        print(f"  Threshold: {self.confidence_threshold} | TP: {self.tp_atr_mult}x ATR | SL: {self.sl_atr_mult}x ATR")

        test_df = self.load_test_data()

        for i in range(len(test_df) - 1):
            row = test_df.iloc[i]
            atr = row.get('atr_pct_14', np.nan)
            if pd.isna(atr):
                continue

            # Predict using saved feature list
            features = row[self.feature_cols].fillna(0).values.reshape(1, -1)
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            prob_tp = self.model.predict_proba(features)[0, 1]

            if prob_tp < self.confidence_threshold:
                continue

            tp_pct, sl_pct = self.dynamic_tp_sl(atr)
            trade = self.simulate_trade(row['close'], tp_pct, sl_pct, prob_tp, test_df.iloc[i:])

            if trade:
                self.trades.append(trade)
                self.capital = trade['capital_after']
                self.equity_curve.append({
                    'timestamp': test_df.index[i],
                    'capital': self.capital
                })

        return self._metrics()

    def _metrics(self):
        if not self.trades:
            return {'error': 'No trades', 'crypto': self.crypto}

        df = pd.DataFrame(self.trades)
        wins = df[df['result'] == 'WIN']
        losses = df[df['result'] == 'LOSE']
        n = len(df)
        w = len(wins)

        total_roi = (self.capital / self.initial_capital - 1) * 100
        pf = wins['pnl'].sum() / abs(losses['pnl'].sum()) if len(losses) > 0 else float('inf')

        max_dd = 0
        if self.equity_curve:
            eq = pd.DataFrame(self.equity_curve)
            eq['peak'] = eq['capital'].cummax()
            max_dd = ((eq['capital'] - eq['peak']) / eq['peak'] * 100).min()

        return {
            'crypto': self.crypto.upper(),
            'total_trades': n, 'wins': w, 'losses': n - w,
            'win_rate': w / n * 100,
            'total_roi': total_roi, 'total_pnl': df['pnl'].sum(),
            'profit_factor': pf, 'max_drawdown': max_dd,
            'avg_tp_pct': float(df['tp_pct'].mean()),
            'avg_sl_pct': float(df['sl_pct'].mean()),
            'avg_hold': float(df['hold'].mean()),
            'avg_confidence': float(df['prob_tp'].mean())
        }, df


def backtest_all():
    print("\n" + "=" * 80)
    print("V12 BACKTEST - Dynamic ATR TP/SL at Execution")
    print("=" * 80)

    cryptos = ['btc', 'eth', 'sol']
    results = {}
    trades = {}

    for c in cryptos:
        try:
            engine = V12BacktestEngine(c)
            r = engine.run_backtest()
            if isinstance(r, dict) and 'error' in r:
                results[c] = r
            else:
                results[c], trades[c] = r
        except Exception as e:
            print(f"\n  ERROR {c}: {e}")
            import traceback
            traceback.print_exc()

    # Load V11 for comparison
    v11_path = PROJECT_ROOT / 'results' / 'backtest_2026_metrics.json'
    v11 = {}
    if v11_path.exists():
        with open(v11_path) as f:
            v11 = json.load(f)

    # Comparison table
    print(f"\n\n{'='*80}")
    print("V12 vs V11 COMPARISON")
    print("=" * 80)
    print(f"\n{'Crypto':<7} {'Ver':<5} {'Trades':>7} {'Win%':>7} {'ROI':>9} {'PF':>6} {'MaxDD':>8} {'AvgTP':>7} {'AvgSL':>7}")
    print("-" * 68)

    for c in cryptos:
        # V11
        if c in v11 and 'error' not in v11[c]:
            v = v11[c]
            print(f"{c.upper():<7} {'V11':<5} {v.get('total_trades',0):>7} "
                  f"{v.get('win_rate',0):>6.1f}% {v.get('total_roi',0):>+8.2f}% "
                  f"{v.get('profit_factor',0):>5.2f} {v.get('max_drawdown',0):>7.2f}% "
                  f"{'1.50%':>7} {'0.75%':>7}")

        # V12
        if c in results and 'error' not in results[c]:
            r = results[c]
            print(f"{'':>7} {'V12':<5} {r['total_trades']:>7} "
                  f"{r['win_rate']:>6.1f}% {r['total_roi']:>+8.2f}% "
                  f"{r['profit_factor']:>5.2f} {r['max_drawdown']:>7.2f}% "
                  f"{r['avg_tp_pct']:>6.2f}% {r['avg_sl_pct']:>6.2f}%")

            if c in v11 and 'error' not in v11[c]:
                d_wr = r['win_rate'] - v11[c].get('win_rate', 0)
                d_roi = r['total_roi'] - v11[c].get('total_roi', 0)
                print(f"{'':>7} {'DELTA':<5} {'':>7} {d_wr:>+6.1f}% {d_roi:>+8.2f}%")
        print()

    # Trade details
    print(f"{'='*80}")
    print("TRADE DETAILS")
    print("=" * 80)

    for c in cryptos:
        if c not in trades:
            continue
        tdf = trades[c]
        print(f"\n{c.upper()} - {len(tdf)} trades:")
        for i, t in tdf.iterrows():
            icon = "WIN " if t['result'] == 'WIN' else "LOSE"
            print(f"  {i+1:2d}. [{icon}] {t['entry_time'].strftime('%Y-%m-%d')} -> "
                  f"{t['exit_time'].strftime('%Y-%m-%d')} | "
                  f"P&L: ${t['pnl']:+7.2f} ({t['pnl_pct']:+.2f}%) | "
                  f"TP:{t['tp_pct']:.2f}% SL:{t['sl_pct']:.2f}% | "
                  f"Conf:{t['prob_tp']:.1%} | {t['hold']}d")

    # Save
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / 'backtest_v12_metrics.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    for c, tdf in trades.items():
        tdf.to_csv(results_dir / f'backtest_v12_{c}_trades.csv')

    print(f"\nSaved to {results_dir}")
    return results, trades


if __name__ == '__main__':
    backtest_all()
