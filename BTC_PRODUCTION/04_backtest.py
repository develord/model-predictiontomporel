"""
BTC PyTorch Backtesting Script - Realistic Intraday
====================================================
Backtest PyTorch model on Q1 2026 data with 15min candle monitoring.

Process:
1. Predict direction on 1D close (daily signal)
2. Download 15min candles from Binance for intraday monitoring
3. Monitor TP/SL hit on each 15min candle
4. Exit at EOD if no barrier hit

Usage:
    python 04_backtest.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import joblib
import json
import ccxt
import logging
from datetime import timedelta

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR / 'scripts'))

from direction_prediction_model import DirectionPredictionModel, LightweightDirectionModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = BASE_DIR / 'data' / 'cache'
MODEL_DIR = BASE_DIR / 'models'
RESULTS_DIR = BASE_DIR / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Trading parameters
TP_PCT = 0.015  # 1.5%
SL_PCT = 0.0075  # 0.75%
POSITION_SIZE = 0.95
TRADING_FEE = 0.001  # 0.1%
SLIPPAGE = 0.0005  # 0.05%
CONFIDENCE_THRESHOLD = 0.55

BACKTEST_START = '2026-01-01'
BACKTEST_END = '2026-03-24'


def load_model():
    """Load model - auto-detect type from checkpoint"""
    model_file = MODEL_DIR / 'BTC_direction_model.pt'
    if not model_file.exists():
        logger.error(f"Model not found: {model_file}")
        return None, None

    checkpoint = torch.load(model_file, map_location='cpu', weights_only=False)

    model_type = checkpoint.get('model_type', 'standard') if isinstance(checkpoint, dict) else 'standard'
    feature_dim = checkpoint.get('feature_dim', 90) if isinstance(checkpoint, dict) else 90
    seq_len = checkpoint.get('sequence_length', 60) if isinstance(checkpoint, dict) else 60

    if model_type == 'lightweight':
        model = LightweightDirectionModel(feature_dim=feature_dim, sequence_length=seq_len, hidden_dim=128, num_layers=2, dropout=0.3)
    else:
        model = DirectionPredictionModel(feature_dim=feature_dim, sequence_length=seq_len)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    logger.info(f"Model loaded: {model_type} (seq_len={seq_len}, features={feature_dim})")
    return model, seq_len


def fetch_15min_data(start_date, end_date):
    """Fetch 15min candles from Binance for intraday TP/SL monitoring"""
    cache_file = DATA_DIR / 'btc_15min_q1_2026.csv'

    if cache_file.exists():
        df = pd.read_csv(cache_file, parse_dates=['timestamp'])
        df.set_index('timestamp', inplace=True)
        logger.info(f"15min data loaded from cache: {len(df)} candles")
        return df

    logger.info("Fetching 15min candles from Binance...")

    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'}
    })

    start_ts = int(pd.Timestamp(start_date).timestamp() * 1000)
    end_ts = int(pd.Timestamp(end_date).timestamp() * 1000)

    all_candles = []
    current_ts = start_ts

    while current_ts < end_ts:
        try:
            candles = exchange.fetch_ohlcv('BTC/USDT', '15m', since=current_ts, limit=1000)
            if not candles:
                break
            all_candles.extend(candles)
            current_ts = candles[-1][0] + 1
            logger.info(f"  Fetched {len(all_candles)} candles...")
        except Exception as e:
            logger.warning(f"  Fetch error: {e}")
            break

    df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[(df.index >= start_date) & (df.index < end_date)]

    # Cache for reuse
    df.to_csv(cache_file)
    logger.info(f"15min data: {len(df)} candles ({df.index[0]} to {df.index[-1]})")

    return df


def simulate_trade_intraday(entry_price, entry_date, df_15m, max_hold_days=10):
    """
    Simulate trade using 15min candles for realistic TP/SL execution.
    Entry at 1D close, monitor next days via 15min candles.
    """
    tp_price = entry_price * (1 + TP_PCT)
    sl_price = entry_price * (1 - SL_PCT)

    # Monitor from next day onwards (up to max_hold_days)
    for day_offset in range(1, max_hold_days + 1):
        check_day = entry_date + timedelta(days=day_offset)
        check_day_end = check_day + timedelta(days=1)

        intraday = df_15m[(df_15m.index >= check_day) & (df_15m.index < check_day_end)]

        if len(intraday) == 0:
            continue

        for ts, candle in intraday.iterrows():
            # Check TP first
            if candle['high'] >= tp_price:
                duration = (ts - entry_date).total_seconds() / 60
                return {
                    'exit_type': 'TP',
                    'exit_price': tp_price * (1 - SLIPPAGE),
                    'exit_time': ts,
                    'pnl_pct': TP_PCT * 100,
                    'duration_minutes': duration
                }

            # Check SL
            if candle['low'] <= sl_price:
                duration = (ts - entry_date).total_seconds() / 60
                return {
                    'exit_type': 'SL',
                    'exit_price': sl_price * (1 - SLIPPAGE),
                    'exit_time': ts,
                    'pnl_pct': -SL_PCT * 100,
                    'duration_minutes': duration
                }

    # Max hold reached - exit at last available close
    last_check = entry_date + timedelta(days=max_hold_days)
    remaining = df_15m[df_15m.index <= last_check]
    if len(remaining) > 0:
        final_price = remaining.iloc[-1]['close']
        pnl_pct = (final_price / entry_price - 1) * 100
        return {
            'exit_type': 'EOD',
            'exit_price': final_price,
            'exit_time': remaining.index[-1],
            'pnl_pct': pnl_pct,
            'duration_minutes': max_hold_days * 1440
        }

    return {
        'exit_type': 'EOD',
        'exit_price': entry_price,
        'exit_time': entry_date + timedelta(days=max_hold_days),
        'pnl_pct': 0.0,
        'duration_minutes': max_hold_days * 1440
    }


def backtest():
    """Run realistic backtest on Q1 2026 with 15min candle monitoring"""
    logger.info(f"\n{'='*70}")
    logger.info(f"BTC PYTORCH BACKTEST - REALISTIC INTRADAY (Q1 2026)")
    logger.info(f"{'='*70}\n")

    model, seq_len = load_model()
    if model is None:
        return

    # Load 1D features for prediction
    df = pd.read_csv(DATA_DIR / 'btc_features.csv')
    df['date'] = pd.to_datetime(df['date'])

    with open(BASE_DIR / 'required_features.json', 'r') as f:
        feature_cols = json.load(f)

    # Load scaler
    scaler = None
    scaler_path = MODEL_DIR / 'feature_scaler.joblib'
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        logger.info("Feature scaler loaded")

    # Prepare full feature matrix
    df_wide = df[df['date'] >= '2025-01-01'].copy()
    all_features = df_wide[feature_cols].fillna(0).values
    if scaler is not None:
        all_features = np.nan_to_num(scaler.transform(all_features), nan=0.0, posinf=0.0, neginf=0.0)

    df_test = df_wide[(df_wide['date'] >= BACKTEST_START) & (df_wide['date'] <= BACKTEST_END)].copy()
    test_start_idx = len(df_wide) - len(df_test)

    logger.info(f"Test period: {len(df_test)} days ({BACKTEST_START} to {BACKTEST_END})")
    logger.info(f"Features: {len(feature_cols)}, seq_len={seq_len}")

    # Fetch 15min candles
    df_15m = fetch_15min_data(BACKTEST_START, BACKTEST_END)

    # Trading simulation
    capital = 1000.0
    trades = []
    in_position = False

    logger.info(f"\nConfidence threshold: {CONFIDENCE_THRESHOLD}")
    logger.info(f"TP: {TP_PCT:.1%} | SL: {SL_PCT:.2%}")
    logger.info(f"Initial capital: ${capital:.2f}")
    logger.info(f"15min candle monitoring for TP/SL execution\n")

    for i in range(len(df_test)):
        wide_idx = test_start_idx + i
        row = df_test.iloc[i]

        if wide_idx < seq_len:
            continue

        if not in_position:
            # Predict using 1D features
            seq = all_features[wide_idx-seq_len:wide_idx]
            X = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                direction, confidence = model.predict_direction(X)

            d = direction.item()
            c = confidence.item()

            if d == 1 and c > CONFIDENCE_THRESHOLD:
                entry_price = row['close'] * (1 + SLIPPAGE)
                entry_date = row['date']
                position_value = capital * POSITION_SIZE
                shares = position_value / entry_price
                capital -= position_value + (position_value * TRADING_FEE)
                in_position = True

                # Simulate trade with 15min candles
                result = simulate_trade_intraday(entry_price, entry_date, df_15m)

                exit_value = shares * result['exit_price']
                capital += exit_value - (exit_value * TRADING_FEE)

                pnl = exit_value - (shares * entry_price)
                actual_pnl_pct = (result['exit_price'] / entry_price - 1) * 100

                trades.append({
                    'entry_date': entry_date,
                    'entry_price': entry_price,
                    'confidence': c,
                    'exit_type': result['exit_type'],
                    'exit_price': result['exit_price'],
                    'exit_time': result['exit_time'],
                    'pnl_pct': actual_pnl_pct,
                    'pnl_usd': pnl,
                    'duration_min': result['duration_minutes']
                })

                dur_h = result['duration_minutes'] / 60
                logger.info(f"  {entry_date.date()} | ENTRY @ ${entry_price:.0f} (Conf: {c:.1%}) | "
                           f"EXIT {result['exit_type']} @ ${result['exit_price']:.0f} | "
                           f"PnL: {actual_pnl_pct:+.2f}% | Duration: {dur_h:.1f}h")

                in_position = False  # Trade completed

    # Results
    logger.info(f"\n{'='*70}")
    logger.info(f"BACKTEST RESULTS - REALISTIC INTRADAY")
    logger.info(f"{'='*70}\n")

    if trades:
        trades_df = pd.DataFrame(trades)
        total_return = (capital / 1000 - 1) * 100

        tp_trades = trades_df[trades_df['exit_type'] == 'TP']
        sl_trades = trades_df[trades_df['exit_type'] == 'SL']
        eod_trades = trades_df[trades_df['exit_type'] == 'EOD']

        wins = len(tp_trades)
        losses = len(sl_trades)
        eod_count = len(eod_trades)
        win_rate = wins / len(trades) * 100 if trades else 0

        logger.info(f"Total Trades: {len(trades)}")
        logger.info(f"  TP (wins):  {wins} ({wins/len(trades)*100:.1f}%)")
        logger.info(f"  SL (losses): {losses} ({losses/len(trades)*100:.1f}%)")
        logger.info(f"  EOD (neutral): {eod_count} ({eod_count/len(trades)*100:.1f}%)")
        logger.info(f"\nWin Rate: {win_rate:.2f}%")
        logger.info(f"Total Return: {total_return:+.2f}%")
        logger.info(f"Final Capital: ${capital:.2f}")

        if len(tp_trades) > 0:
            logger.info(f"\nAvg TP Duration: {tp_trades['duration_min'].mean()/60:.1f}h")
        if len(sl_trades) > 0:
            logger.info(f"Avg SL Duration: {sl_trades['duration_min'].mean()/60:.1f}h")
        logger.info(f"Avg Total Duration: {trades_df['duration_min'].mean()/60:.1f}h")

        # Save results
        trades_df.to_csv(RESULTS_DIR / 'btc_backtest_trades.csv', index=False)

        summary = {
            'backtest_type': 'realistic_intraday_15min',
            'period': f'{BACKTEST_START} to {BACKTEST_END}',
            'total_trades': len(trades),
            'wins': wins,
            'losses': losses,
            'eod': eod_count,
            'win_rate': win_rate,
            'total_return_pct': total_return,
            'initial_capital': 1000,
            'final_capital': round(capital, 2),
            'confidence_threshold': CONFIDENCE_THRESHOLD,
            'tp_pct': TP_PCT,
            'sl_pct': SL_PCT
        }
        with open(RESULTS_DIR / 'btc_backtest_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"\nResults saved to {RESULTS_DIR}")
    else:
        logger.info("No trades executed")

    logger.info(f"\n{'='*70}")


if __name__ == "__main__":
    backtest()
