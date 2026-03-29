"""
V11 Backtesting with Intraday 15min Monitoring
===============================================

Two-Phase Backtesting with Model Retraining:
- Phase 1: RETRAIN on 2018-2023, TEST on 2023
- Phase 2: RETRAIN on 2018-Jul2025, TEST on Aug2025-Today

Key Features:
- Model is RETRAINED for each phase with specific date ranges
- Uses 1D candles for prediction (daily close)
- Monitors 15min candles intraday for TP/SL execution
- Real intraday price movement simulation

Usage:
    python backtesting/backtest_v11_intraday.py --crypto btc --mode baseline --phase 1
    python backtesting/backtest_v11_intraday.py --crypto eth --mode optuna --phase 2
"""

import sys
import argparse
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta
import ccxt
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import xgboost as xgb
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent))


class IntradayBacktester:
    """Backtest V11 models with 15min intraday TP/SL monitoring"""

    def __init__(self, crypto: str, mode: str, phase: int):
        """
        Args:
            crypto: 'btc', 'eth', or 'sol'
            mode: 'baseline', 'optuna', or 'top50'
            phase: 1 or 2
        """
        self.crypto = crypto.lower()
        self.mode = mode
        self.phase = phase

        # Trading parameters (from triple barrier)
        self.tp_pct = 0.02  # 2% take profit
        self.sl_pct = 0.01  # 1% stop loss
        self.threshold = 0.60  # P(TP) threshold for trade entry

        # Phase periods - IMPORTANT: Model is RETRAINED for each phase
        if phase == 1:
            self.train_start = '2018-01-01'
            self.train_end = '2023-01-01'
            self.test_start = '2023-01-01'
            self.test_end = '2024-01-01'
        else:  # phase == 2
            self.train_start = '2018-01-01'
            self.train_end = '2025-07-01'
            self.test_start = '2025-08-01'
            self.test_end = datetime.now().strftime('%Y-%m-%d')

        self.model = None
        self.feature_cols = None
        self.results_dir = Path(__file__).parent.parent / 'backtesting' / 'results'
        self.results_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*80}")
        print(f"V11 INTRADAY BACKTEST - Phase {phase}")
        print(f"{'='*80}")
        print(f"Crypto: {crypto.upper()} | Mode: {mode}")
        print(f"\nMODEL WILL BE RETRAINED FOR THIS PHASE:")
        print(f"   Training Period: {self.train_start} to {self.train_end}")
        print(f"   Testing Period:  {self.test_start} to {self.test_end}")
        print(f"{'='*80}\n")


    def load_1d_data(self) -> pd.DataFrame:
        """Load 1D candle data for model predictions"""
        cache_file = Path(__file__).parent.parent / 'data' / 'cache' / f'{self.crypto}_multi_tf_merged.csv'

        if not cache_file.exists():
            raise FileNotFoundError(f"Data not found: {cache_file}")

        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        print(f"[1D Data] Loaded {len(df)} daily candles ({df.index[0].date()} to {df.index[-1].date()})")

        return df


    def get_hyperparameters(self, scale_pos_weight: float) -> dict:
        """
        Get hyperparameters based on mode (baseline, optuna, top50)

        Args:
            scale_pos_weight: Class imbalance weight

        Returns:
            dict: XGBoost hyperparameters
        """
        if self.mode == 'baseline' or self.mode == 'top50':
            return self.get_baseline_params(scale_pos_weight)
        elif self.mode == 'optuna':
            # Optuna-optimized parameters per crypto
            optuna_configs = {
                'btc': {
                    'objective': 'binary:logistic',
                    'eval_metric': 'auc',
                    'max_depth': 5,
                    'learning_rate': 0.04,
                    'n_estimators': 300,
                    'gamma': 1.5,
                    'min_child_weight': 3,
                    'subsample': 0.85,
                    'colsample_bytree': 0.85,
                    'colsample_bylevel': 0.8,
                    'reg_alpha': 0.3,
                    'reg_lambda': 1.5,
                    'scale_pos_weight': scale_pos_weight,
                    'random_state': 42,
                    'tree_method': 'hist'
                },
                'eth': {
                    'objective': 'binary:logistic',
                    'eval_metric': 'auc',
                    'max_depth': 7,
                    'learning_rate': 0.03,
                    'n_estimators': 350,
                    'gamma': 1.0,
                    'min_child_weight': 2,
                    'subsample': 0.9,
                    'colsample_bytree': 0.9,
                    'colsample_bylevel': 0.85,
                    'reg_alpha': 0.2,
                    'reg_lambda': 1.2,
                    'scale_pos_weight': scale_pos_weight,
                    'random_state': 42,
                    'tree_method': 'hist'
                },
                'sol': {
                    'objective': 'binary:logistic',
                    'eval_metric': 'auc',
                    'max_depth': 5,
                    'learning_rate': 0.0337,
                    'n_estimators': 392,
                    'gamma': 0.8069,
                    'min_child_weight': 8,
                    'subsample': 0.85,
                    'colsample_bytree': 0.85,
                    'colsample_bylevel': 0.8,
                    'reg_alpha': 0.25,
                    'reg_lambda': 1.3,
                    'scale_pos_weight': scale_pos_weight,
                    'random_state': 42,
                    'tree_method': 'hist'
                }
            }
            # Fallback to baseline if crypto not found
            if self.crypto not in optuna_configs:
                return self.get_baseline_params(scale_pos_weight)
            return optuna_configs[self.crypto]

        # Default fallback
        return self.get_baseline_params(scale_pos_weight)


    def get_baseline_params(self, scale_pos_weight: float) -> dict:
        """Get baseline hyperparameters"""
        return {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 200,
            'gamma': 2,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'scale_pos_weight': scale_pos_weight,
            'random_state': 42,
            'tree_method': 'hist'
        }


    def train_model(self):
        """
        RETRAIN model on specified training period for this phase
        This ensures the model is trained on the correct data for each phase
        """
        print(f"\n{'='*80}")
        print(f"RETRAINING MODEL - Phase {self.phase}")
        print(f"{'='*80}")

        # Load data
        df = self.load_1d_data()

        # Prepare features
        exclude_cols = [
            'open', 'high', 'low', 'close', 'volume',
            'label_class', 'label_numeric',
            'price_target_pct', 'future_price',
            'triple_barrier_label'
        ]

        all_features = [col for col in df.columns if col not in exclude_cols]

        # For top50 mode, load feature importance from baseline
        if self.mode == 'top50':
            baseline_stats_file = Path(__file__).parent.parent / 'models' / f'{self.crypto}_v11_baseline_stats.json'

            if baseline_stats_file.exists():
                with open(baseline_stats_file, 'r') as f:
                    baseline_stats = json.load(f)

                top_features = baseline_stats.get('top_features', [])[:50]
                top_feature_names = [f['feature'] for f in top_features]
                self.feature_cols = [f for f in top_feature_names if f in all_features]
                print(f"Using TOP 50 features from baseline model")
            else:
                print(f"Warning: Baseline stats not found, using all features")
                self.feature_cols = all_features
        else:
            self.feature_cols = all_features

        # Filter to training period
        df_train = df[(df.index >= self.train_start) & (df.index < self.train_end)]
        df_train = df_train[df_train['triple_barrier_label'].notna()].copy()

        print(f"\nTraining Period: {df_train.index[0].date()} to {df_train.index[-1].date()}")
        print(f"Training Samples: {len(df_train)}")
        print(f"Features: {len(self.feature_cols)}")

        # Prepare X, y
        X_train = df_train[self.feature_cols].fillna(0).values
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)

        y_train = (df_train['triple_barrier_label'].values == 1).astype(int)

        # Calculate class weight
        n_neg = np.sum(y_train == 0)
        n_pos = np.sum(y_train == 1)
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

        print(f"\nClass Distribution:")
        print(f"  SL (0): {n_neg} ({n_neg/len(y_train)*100:.1f}%)")
        print(f"  TP (1): {n_pos} ({n_pos/len(y_train)*100:.1f}%)")
        print(f"  Scale weight: {scale_pos_weight:.2f}")

        # Get hyperparameters
        params = self.get_hyperparameters(scale_pos_weight)

        print(f"\nHyperparameters ({self.mode}):")
        print(f"  max_depth: {params['max_depth']}")
        print(f"  learning_rate: {params['learning_rate']}")
        print(f"  n_estimators: {params['n_estimators']}")
        print(f"  gamma: {params['gamma']}")

        # Train
        print("\nTraining XGBoost model...")
        self.model = xgb.XGBClassifier(**params, verbosity=0)
        self.model.fit(X_train, y_train)

        print("Model trained successfully!")

        # Save model
        model_file = self.results_dir / f'{self.crypto}_phase{self.phase}_{self.mode}_model.joblib'
        joblib.dump(self.model, model_file)
        print(f"Model saved: {model_file}")

        # Save feature list
        features_file = self.results_dir / f'{self.crypto}_phase{self.phase}_{self.mode}_features.json'
        with open(features_file, 'w') as f:
            json.dump({'features': self.feature_cols}, f, indent=2)
        print(f"Features saved: {features_file}")


    def fetch_15min_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch 15min candles from exchange for intraday monitoring

        Args:
            start_date: Start date for 15min data
            end_date: End date for 15min data

        Returns:
            DataFrame with 15min OHLCV data
        """
        print(f"\n[15min Data] Fetching from exchange...")

        # Symbol mapping
        symbols = {
            'btc': 'BTC/USDT',
            'eth': 'ETH/USDT',
            'sol': 'SOL/USDT'
        }

        try:
            exchange = ccxt.binance({
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'}
            })

            symbol = symbols[self.crypto]
            timeframe = '15m'

            # Convert dates to timestamps
            start_ts = int(pd.Timestamp(start_date).timestamp() * 1000)
            end_ts = int(pd.Timestamp(end_date).timestamp() * 1000)

            all_candles = []
            current_ts = start_ts

            while current_ts < end_ts:
                try:
                    candles = exchange.fetch_ohlcv(
                        symbol,
                        timeframe,
                        since=current_ts,
                        limit=1000
                    )

                    if not candles:
                        break

                    all_candles.extend(candles)
                    current_ts = candles[-1][0] + 1

                    print(f"  Fetched {len(all_candles)} candles...", end='\r')

                except Exception as e:
                    print(f"\n  Warning: {e}")
                    break

            # Convert to DataFrame
            df_15m = pd.DataFrame(
                all_candles,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )

            df_15m['timestamp'] = pd.to_datetime(df_15m['timestamp'], unit='ms')
            df_15m.set_index('timestamp', inplace=True)

            # Filter to exact date range
            df_15m = df_15m[(df_15m.index >= start_date) & (df_15m.index < end_date)]

            print(f"\n[15min Data] Loaded {len(df_15m)} candles")
            print(f"  Period: {df_15m.index[0]} to {df_15m.index[-1]}")

            return df_15m

        except Exception as e:
            print(f"\n[15min Data] Warning: {e}")
            print("Using simulated 15min data from 1D candles...")
            return self._simulate_15min_data(start_date, end_date)


    def _simulate_15min_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Simulate 15min candles from 1D data if exchange fetch fails
        Uses realistic intraday volatility simulation
        """
        df_1d = self.load_1d_data()
        df_1d = df_1d[(df_1d.index >= start_date) & (df_1d.index < end_date)]

        all_15m = []

        for idx, row in df_1d.iterrows():
            # Create 96 15min candles per day (24h * 4)
            date = idx
            daily_range = row['high'] - row['low']
            daily_volatility = daily_range / row['close'] if row['close'] > 0 else 0.01

            for i in range(96):
                timestamp = date + timedelta(minutes=15*i)

                # Realistic price path simulation
                progress = i / 96

                # Random walk with drift towards close
                drift = (row['close'] - row['open']) * progress
                noise = np.random.normal(0, daily_volatility * 0.1) * row['open']
                price = row['open'] + drift + noise

                # Constrain within daily high/low
                price = max(min(price, row['high']), row['low'])

                # Simulate high/low for this 15min candle
                intraday_vol = daily_volatility * 0.05
                high = price * (1 + abs(np.random.normal(0, intraday_vol)))
                low = price * (1 - abs(np.random.normal(0, intraday_vol)))

                # Ensure within daily bounds
                high = min(high, row['high'])
                low = max(low, row['low'])

                all_15m.append({
                    'timestamp': timestamp,
                    'open': price,
                    'high': high,
                    'low': low,
                    'close': price,
                    'volume': row['volume'] / 96
                })

        df_15m = pd.DataFrame(all_15m)
        df_15m.set_index('timestamp', inplace=True)

        print(f"[15min Data] Simulated {len(df_15m)} candles")

        return df_15m


    def simulate_trade_intraday(self, entry_price: float, entry_date: pd.Timestamp,
                                df_15m: pd.DataFrame) -> dict:
        """
        Simulate trade execution using 15min candles
        Trade is entered at today's close, monitored next day via 15min candles

        Args:
            entry_price: Entry price (1D close)
            entry_date: Entry date
            df_15m: 15min candle data

        Returns:
            dict with trade result
        """
        tp_price = entry_price * (1 + self.tp_pct)
        sl_price = entry_price * (1 - self.sl_pct)

        # Get next day's 15min candles (trade monitoring day)
        next_day = entry_date + timedelta(days=1)
        next_day_end = next_day + timedelta(days=1)

        # Filter 15min candles for next trading day
        intraday_candles = df_15m[(df_15m.index >= next_day) & (df_15m.index < next_day_end)]

        if len(intraday_candles) == 0:
            # No intraday data, use daily close
            return {
                'exit_type': 'eod',
                'exit_price': entry_price,
                'exit_time': next_day_end,
                'pnl_pct': 0.0,
                'duration_minutes': 1440
            }

        # Check each 15min candle for TP/SL hit
        for timestamp, candle in intraday_candles.iterrows():
            high = candle['high']
            low = candle['low']

            # Check TP hit first (optimistic fill)
            if high >= tp_price:
                duration = (timestamp - next_day).total_seconds() / 60
                return {
                    'exit_type': 'tp',
                    'exit_price': tp_price,
                    'exit_time': timestamp,
                    'pnl_pct': self.tp_pct,
                    'duration_minutes': duration
                }

            # Check SL hit
            if low <= sl_price:
                duration = (timestamp - next_day).total_seconds() / 60
                return {
                    'exit_type': 'sl',
                    'exit_price': sl_price,
                    'exit_time': timestamp,
                    'pnl_pct': -self.sl_pct,
                    'duration_minutes': duration
                }

        # End of day - no TP/SL hit, exit at close
        final_price = intraday_candles.iloc[-1]['close']
        pnl_pct = (final_price - entry_price) / entry_price

        return {
            'exit_type': 'eod',
            'exit_price': final_price,
            'exit_time': intraday_candles.index[-1],
            'pnl_pct': pnl_pct,
            'duration_minutes': 1440
        }


    def run_backtest(self):
        """Run full backtest with intraday monitoring"""
        print(f"\n{'='*80}")
        print(f"RUNNING BACKTEST - Phase {self.phase}")
        print(f"{'='*80}")

        # Load 1D data
        df_1d = self.load_1d_data()
        df_test = df_1d[(df_1d.index >= self.test_start) & (df_1d.index < self.test_end)]

        print(f"\nTest Period: {len(df_test)} days ({self.test_start} to {self.test_end})")

        # Fetch 15min data
        df_15m = self.fetch_15min_data(self.test_start, self.test_end)

        # Prepare features
        X_test = df_test[self.feature_cols].fillna(0).values
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

        # Predict
        print("\nGenerating predictions...")
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        print(f"P(TP) Distribution: min={y_pred_proba.min():.3f}, max={y_pred_proba.max():.3f}, mean={y_pred_proba.mean():.3f}")

        # Simulate trades
        trades = []

        print(f"\nSimulating trades (threshold P(TP) >= {self.threshold})...")
        print("Monitoring with 15min candles for TP/SL execution...\n")

        for i, (date, row) in enumerate(df_test.iterrows()):
            prob_tp = y_pred_proba[i]

            # Entry signal
            if prob_tp >= self.threshold:
                entry_price = row['close']

                # Simulate intraday execution
                result = self.simulate_trade_intraday(entry_price, date, df_15m)

                trades.append({
                    'entry_date': date,
                    'entry_price': entry_price,
                    'prob_tp': prob_tp,
                    'exit_type': result['exit_type'],
                    'exit_price': result['exit_price'],
                    'exit_time': result['exit_time'],
                    'pnl_pct': result['pnl_pct'],
                    'duration_minutes': result['duration_minutes']
                })

                print(f"  Trade {len(trades):3d}: {date.date()} | P(TP)={prob_tp:.3f} | "
                      f"{result['exit_type'].upper():3s} @ {result['duration_minutes']:4.0f}min | "
                      f"PnL: {result['pnl_pct']*100:+6.2f}%")

        print(f"\nTotal trades executed: {len(trades)}")

        return pd.DataFrame(trades)


    def calculate_metrics(self, df_trades: pd.DataFrame) -> dict:
        """Calculate backtest performance metrics"""
        if len(df_trades) == 0:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_pnl_pct': 0,
                'total_return_pct': 0,
                'tp_count': 0,
                'sl_count': 0,
                'eod_count': 0
            }

        # Basic stats
        total_trades = len(df_trades)

        # Win rate
        wins = len(df_trades[df_trades['pnl_pct'] > 0])
        win_rate = wins / total_trades if total_trades > 0 else 0

        # PnL stats
        avg_pnl = df_trades['pnl_pct'].mean()
        total_return = df_trades['pnl_pct'].sum()
        best_trade = df_trades['pnl_pct'].max()
        worst_trade = df_trades['pnl_pct'].min()

        # TP/SL breakdown
        tp_count = len(df_trades[df_trades['exit_type'] == 'tp'])
        sl_count = len(df_trades[df_trades['exit_type'] == 'sl'])
        eod_count = len(df_trades[df_trades['exit_type'] == 'eod'])

        # Duration stats
        avg_duration = df_trades['duration_minutes'].mean()
        median_duration = df_trades['duration_minutes'].median()

        metrics = {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_pnl_pct': avg_pnl * 100,
            'total_return_pct': total_return * 100,
            'best_trade_pct': best_trade * 100,
            'worst_trade_pct': worst_trade * 100,
            'tp_count': tp_count,
            'sl_count': sl_count,
            'eod_count': eod_count,
            'avg_duration_minutes': avg_duration,
            'median_duration_minutes': median_duration,
            'tp_rate': tp_count / total_trades if total_trades > 0 else 0,
            'sl_rate': sl_count / total_trades if total_trades > 0 else 0,
            'eod_rate': eod_count / total_trades if total_trades > 0 else 0
        }

        return metrics


    def print_results(self, metrics: dict, df_trades: pd.DataFrame):
        """Print backtest results"""
        print(f"\n{'='*80}")
        print(f"BACKTEST RESULTS - Phase {self.phase}")
        print(f"{'='*80}")
        print(f"Crypto: {self.crypto.upper()} | Mode: {self.mode}")
        print(f"Period: {self.test_start} to {self.test_end}")

        print(f"\n{'-'*80}")
        print("TRADE STATISTICS")
        print(f"{'-'*80}")
        print(f"  Total Trades:       {metrics['total_trades']}")
        print(f"  Win Rate:           {metrics['win_rate']*100:.2f}%")
        print(f"  Average PnL:        {metrics['avg_pnl_pct']:+.2f}%")
        print(f"  Total Return:       {metrics['total_return_pct']:+.2f}%")
        print(f"  Best Trade:         {metrics['best_trade_pct']:+.2f}%")
        print(f"  Worst Trade:        {metrics['worst_trade_pct']:+.2f}%")

        print(f"\n{'-'*80}")
        print("EXIT TYPE BREAKDOWN")
        print(f"{'-'*80}")
        print(f"  TP Hit:             {metrics['tp_count']:3d} ({metrics['tp_rate']*100:5.1f}%) - Target reached!")
        print(f"  SL Hit:             {metrics['sl_count']:3d} ({metrics['sl_rate']*100:5.1f}%) - Stop loss triggered")
        print(f"  End of Day:         {metrics['eod_count']:3d} ({metrics['eod_rate']*100:5.1f}%) - No TP/SL hit")

        print(f"\n{'-'*80}")
        print("TIMING ANALYSIS")
        print(f"{'-'*80}")
        print(f"  Avg Duration:       {metrics['avg_duration_minutes']:6.0f} minutes ({metrics['avg_duration_minutes']/60:.1f} hours)")
        print(f"  Median Duration:    {metrics['median_duration_minutes']:6.0f} minutes ({metrics['median_duration_minutes']/60:.1f} hours)")

        if len(df_trades) > 0:
            print(f"\n{'-'*80}")
            print("RECENT TRADES (Last 10)")
            print(f"{'-'*80}")
            recent = df_trades[['entry_date', 'prob_tp', 'exit_type', 'pnl_pct', 'duration_minutes']].tail(10).copy()
            recent['entry_date'] = recent['entry_date'].dt.strftime('%Y-%m-%d')
            recent['pnl_pct'] = recent['pnl_pct'].apply(lambda x: f"{x*100:+.2f}%")
            recent['duration_minutes'] = recent['duration_minutes'].apply(lambda x: f"{x:.0f}min")
            recent['prob_tp'] = recent['prob_tp'].apply(lambda x: f"{x:.3f}")
            print(recent.to_string(index=False))


    def save_results(self, df_trades: pd.DataFrame, metrics: dict):
        """Save backtest results to files"""
        # Save trades
        trades_file = self.results_dir / f'{self.crypto}_phase{self.phase}_{self.mode}_trades.csv'
        df_trades.to_csv(trades_file, index=False)
        print(f"\nTrades saved: {trades_file}")

        # Save metrics
        metrics_file = self.results_dir / f'{self.crypto}_phase{self.phase}_{self.mode}_metrics.json'

        # Add metadata
        full_results = {
            'metadata': {
                'crypto': self.crypto,
                'mode': self.mode,
                'phase': self.phase,
                'train_period': f"{self.train_start} to {self.train_end}",
                'test_period': f"{self.test_start} to {self.test_end}",
                'tp_pct': self.tp_pct,
                'sl_pct': self.sl_pct,
                'threshold': self.threshold
            },
            'metrics': metrics
        }

        with open(metrics_file, 'w') as f:
            json.dump(full_results, f, indent=2)
        print(f"Metrics saved: {metrics_file}")


def main():
    parser = argparse.ArgumentParser(description='V11 Intraday Backtesting with Model Retraining')
    parser.add_argument('--crypto', type=str, required=True, choices=['btc', 'eth', 'sol'])
    parser.add_argument('--mode', type=str, required=True, choices=['baseline', 'optuna', 'top50'])
    parser.add_argument('--phase', type=int, required=True, choices=[1, 2],
                        help='Phase 1: Train 2018-2023, Test 2023 | Phase 2: Train 2018-Jul2025, Test Aug2025-Today')

    args = parser.parse_args()

    # Initialize backtester
    backtester = IntradayBacktester(args.crypto, args.mode, args.phase)

    # ALWAYS retrain model for the phase
    backtester.train_model()

    # Run backtest
    df_trades = backtester.run_backtest()

    # Calculate metrics
    metrics = backtester.calculate_metrics(df_trades)

    # Print results
    backtester.print_results(metrics, df_trades)

    # Save results
    backtester.save_results(df_trades, metrics)

    print(f"\n{'='*80}")
    print("BACKTEST COMPLETE!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
