"""
Backtest Service - Run historical simulations
=============================================
Service pour exécuter des backtests sur des périodes personnalisées
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class BacktestService:
    """Service for running backtests on V11 models"""

    def __init__(self):
        self.models = {}
        self.data_cache = {}
        self.project_root = Path(__file__).parent.parent

    def load_model(self, crypto: str):
        """Load V11 model for given crypto"""
        if crypto in self.models:
            return

        # Convert to short form
        crypto_map = {'bitcoin': 'btc', 'ethereum': 'eth', 'solana': 'sol'}
        short_crypto = crypto_map.get(crypto.lower(), crypto.lower())

        # Load V11 classifier model
        model_file = self.project_root / 'models' / f'{short_crypto}_v11_classifier.joblib'
        if not model_file.exists():
            raise FileNotFoundError(f"Model not found: {model_file}")

        self.models[crypto] = joblib.load(model_file)
        logger.info(f"Loaded V11 model for {crypto}")

    def load_data(self, crypto: str) -> pd.DataFrame:
        """Load historical data for given crypto"""
        if crypto in self.data_cache:
            return self.data_cache[crypto]

        # Convert to short form
        crypto_map = {'bitcoin': 'btc', 'ethereum': 'eth', 'solana': 'sol'}
        short_crypto = crypto_map.get(crypto.lower(), crypto.lower())

        # Load data from cache (same as training project!)
        # Use SHORT form for cache files (btc, eth, sol) not full names
        cache_file = Path(__file__).parent.parent.parent / 'crypto_v10_multi_tf' / 'data' / 'cache' / f'{short_crypto}_multi_tf_merged.csv'
        if not cache_file.exists():
            raise FileNotFoundError(f"Data file not found: {cache_file}")

        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        self.data_cache[crypto] = df
        logger.info(f"Loaded data for {crypto}: {len(df)} candles")

        return df

    def run_backtest(
        self,
        crypto: str,
        start_date: str,
        end_date: str,
        tp_pct: float = 1.5,
        sl_pct: float = 0.75,
        prob_threshold: float = 0.5
    ) -> Dict:
        """
        Run backtest on specified date range

        Args:
            crypto: Cryptocurrency name
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            tp_pct: Take profit percentage
            sl_pct: Stop loss percentage
            prob_threshold: Probability threshold for entry

        Returns:
            Dictionary with metrics and trades
        """
        logger.info(f"Running backtest for {crypto}: {start_date} to {end_date}")

        # Load model and data
        self.load_model(crypto)
        df = self.load_data(crypto)

        # Filter by date range
        try:
            df_test = df.loc[start_date:end_date].copy()
        except Exception as e:
            logger.error(f"Error filtering dates: {e}")
            raise ValueError(f"Invalid date range: {start_date} to {end_date}")

        if len(df_test) == 0:
            raise ValueError(f"No data found for date range {start_date} to {end_date}")

        logger.info(f"Backtest data: {len(df_test)} candles")

        # Get feature columns
        exclude_cols = [
            'open', 'high', 'low', 'close', 'volume',
            'label_class', 'label_numeric', 'price_target_pct',
            'future_price', 'triple_barrier_label'
        ]
        feature_cols = [col for col in df_test.columns if col not in exclude_cols]

        # Prepare features
        X_test = df_test[feature_cols].fillna(0).values
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

        # Predict probabilities
        prob_tp = self.models[crypto].predict_proba(X_test)[:, 1]

        # Add to dataframe
        df_test['prob_tp'] = prob_tp
        df_test['signal'] = (prob_tp > prob_threshold).astype(int)

        # Simulate trades
        trades = self._simulate_trades(df_test, tp_pct, sl_pct)

        # Calculate metrics
        metrics = self._calculate_metrics(trades, df_test, tp_pct, sl_pct, prob_threshold)

        # Format trades for response
        trades_formatted = self._format_trades(trades, df_test)

        return {
            'metrics': metrics,
            'trades': trades_formatted,
            'total_candles': len(df_test),
            'start_date': start_date,
            'end_date': end_date
        }

    def _simulate_trades(self, df: pd.DataFrame, tp_pct: float, sl_pct: float) -> List[Dict]:
        """Simulate trades with fixed TP/SL using high/low intra-candle prices"""
        trades = []
        in_position = False
        entry_idx = None
        entry_price = None
        entry_date = None
        tp_price = None
        sl_price = None

        for idx in range(len(df)):
            row = df.iloc[idx]
            current_date = df.index[idx]

            # Entry signal
            if not in_position and row['signal'] == 1:
                in_position = True
                entry_idx = idx
                entry_price = row['close']
                entry_date = current_date
                # Calculate TP and SL prices
                tp_price = entry_price * (1 + tp_pct / 100)
                sl_price = entry_price * (1 - sl_pct / 100)
                # DON'T continue - check exit on same candle!

            # Check exit conditions using high/low (same as training project)
            if in_position:
                high = row['high']
                low = row['low']

                # Check TP first (priority) - use HIGH
                if high >= tp_price:
                    trades.append({
                        'entry_idx': entry_idx,
                        'exit_idx': idx,
                        'entry_date': entry_date,
                        'exit_date': current_date,
                        'entry_price': entry_price,
                        'exit_price': tp_price,  # Exit at TP price
                        'pnl_pct': tp_pct,
                        'outcome': 'WIN',
                        'bars_held': idx - entry_idx
                    })
                    in_position = False
                    continue

                # Check SL - use LOW
                if low <= sl_price:
                    trades.append({
                        'entry_idx': entry_idx,
                        'exit_idx': idx,
                        'entry_date': entry_date,
                        'exit_date': current_date,
                        'entry_price': entry_price,
                        'exit_price': sl_price,  # Exit at SL price
                        'pnl_pct': -sl_pct,
                        'outcome': 'LOSS',
                        'bars_held': idx - entry_idx
                    })
                    in_position = False
                    continue

        # Close any open position
        if in_position:
            final_price = df.iloc[-1]['close']
            final_date = df.index[-1]
            final_pnl = ((final_price - entry_price) / entry_price) * 100
            trades.append({
                'entry_idx': entry_idx,
                'exit_idx': len(df) - 1,
                'entry_date': entry_date,
                'exit_date': final_date,
                'entry_price': entry_price,
                'exit_price': final_price,
                'pnl_pct': final_pnl,
                'outcome': 'OPEN',
                'bars_held': len(df) - 1 - entry_idx
            })

        return trades

    def _calculate_metrics(
        self,
        trades: List[Dict],
        df: pd.DataFrame,
        tp_pct: float,
        sl_pct: float,
        prob_threshold: float
    ) -> Dict:
        """Calculate performance metrics"""
        if len(trades) == 0:
            return {
                'total_trades': 0,
                'win_trades': 0,
                'loss_trades': 0,
                'open_trades': 0,
                'win_rate': 0,
                'total_roi': 0,
                'avg_trade_roi': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'avg_bars_held': 0,
                'expected_value': 0
            }

        trades_df = pd.DataFrame(trades)

        # Basic stats
        total_trades = len(trades_df)
        win_trades = len(trades_df[trades_df['outcome'] == 'WIN'])
        loss_trades = len(trades_df[trades_df['outcome'] == 'LOSS'])
        open_trades = len(trades_df[trades_df['outcome'] == 'OPEN'])

        win_rate = win_trades / total_trades if total_trades > 0 else 0

        # ROI
        total_roi = trades_df['pnl_pct'].sum()
        avg_trade_roi = trades_df['pnl_pct'].mean()

        # Sharpe ratio (annualized)
        returns = trades_df['pnl_pct'].values
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(365 / 4) if np.std(returns) > 0 else 0

        # Max drawdown
        cumulative_roi = np.cumsum(trades_df['pnl_pct'].values)
        running_max = np.maximum.accumulate(cumulative_roi)
        drawdown = running_max - cumulative_roi
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0

        # Expected value
        ev = (tp_pct * win_rate) + (-sl_pct * (1 - win_rate))

        # Average holding time
        avg_bars_held = trades_df['bars_held'].mean()

        return {
            'total_trades': int(total_trades),
            'win_trades': int(win_trades),
            'loss_trades': int(loss_trades),
            'open_trades': int(open_trades),
            'win_rate': float(win_rate),
            'total_roi': float(total_roi),
            'avg_trade_roi': float(avg_trade_roi),
            'sharpe_ratio': float(sharpe),
            'max_drawdown': float(max_drawdown),
            'avg_bars_held': float(avg_bars_held),
            'expected_value': float(ev),
            'tp_pct': float(tp_pct),
            'sl_pct': float(sl_pct),
            'prob_threshold': float(prob_threshold)
        }

    def _format_trades(self, trades: List[Dict], df: pd.DataFrame) -> List[Dict]:
        """Format trades for API response"""
        formatted = []
        for trade in trades:
            formatted.append({
                'entry_date': trade['entry_date'].strftime('%Y-%m-%d %H:%M:%S'),
                'exit_date': trade['exit_date'].strftime('%Y-%m-%d %H:%M:%S'),
                'entry_price': float(trade['entry_price']),
                'exit_price': float(trade['exit_price']),
                'pnl_pct': float(trade['pnl_pct']),
                'pnl_usd': 0,  # Can calculate based on position size if needed
                'outcome': trade['outcome'],
                'duration_hours': int(trade['bars_held'] * 4)  # Assuming 4H candles
            })
        return formatted


# Global instance
_backtest_service = None


def get_backtest_service() -> BacktestService:
    """Get or create backtest service instance"""
    global _backtest_service
    if _backtest_service is None:
        _backtest_service = BacktestService()
    return _backtest_service
