"""
ETH Backtesting Script
======================
Backtest trained XGBoost model with intelligent signal filtering on Q1 2026

Usage:
    python 04_backtest.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import joblib
import json
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data' / 'cache'
MODEL_DIR = BASE_DIR / 'models'
RESULTS_DIR = BASE_DIR / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CRYPTO = 'ETH'
BACKTEST_START = '2026-01-01'
BACKTEST_END = '2026-03-24'

# Trading parameters
TP_PCT = 0.015  # 1.5%
SL_PCT = 0.0075  # 0.75%
POSITION_SIZE = 0.95  # 95%
TRADING_FEE = 0.001  # 0.1%
SLIPPAGE = 0.0005  # 0.05%

# Intelligent filtering thresholds
MIN_CONFIDENCE = 0.65
MAX_VOLATILITY_1D = 0.05  # 5%
MAX_VOLATILITY_4H = 0.04  # 4%
MAX_VOLATILITY_1W = 0.06  # 6%
MIN_VOLUME_RATIO = 0.8
MIN_ADX = 15
MIN_MOMENTUM_ALIGNMENT = 1  # At least 1/3 timeframes agree (relaxed)

# Bear market filter
MAX_CONSECUTIVE_LOSSES = 2
COOLDOWN_DAYS = 5


def calculate_volatility(df, col='close', window=20):
    """Calculate rolling volatility"""
    returns = df[col].pct_change()
    return returns.rolling(window).std()


def check_momentum_alignment(row):
    """Check if momentum aligns across timeframes"""
    alignments = 0
    if row.get('1d_rsi_14', 50) > 50:
        alignments += 1
    if row.get('4h_rsi_14', 50) > 50:
        alignments += 1
    if row.get('1h_rsi_14', 50) > 50:
        alignments += 1
    return alignments


def intelligent_signal_filter(row, df, idx):
    """5-criteria intelligent filtering system"""

    # 1. Confidence threshold
    if row['confidence'] < MIN_CONFIDENCE:
        return False, "low_confidence"

    # 2. Volatility checks
    vol_1d = calculate_volatility(df.iloc[:idx+1], 'close', 20).iloc[-1]
    if pd.notna(vol_1d) and vol_1d > MAX_VOLATILITY_1D:
        return False, "high_volatility_1d"

    # 3. Volume filter
    if 'volume' in df.columns:
        avg_volume = df['volume'].rolling(20).mean().iloc[idx]
        if pd.notna(avg_volume) and row['volume'] / avg_volume < MIN_VOLUME_RATIO:
            return False, "low_volume"

    # 4. Trend strength (ADX)
    if '1d_adx_14' in row and pd.notna(row['1d_adx_14']):
        if row['1d_adx_14'] < MIN_ADX:
            return False, "weak_trend"

    # 5. Multi-timeframe momentum alignment
    momentum_align = check_momentum_alignment(row)
    if momentum_align < MIN_MOMENTUM_ALIGNMENT:
        return False, "poor_momentum_alignment"

    # 6. Bear market filter - price below SMA20
    if '1d_sma_20' in row.index and pd.notna(row['1d_sma_20']) and row['1d_sma_20'] > 0:
        dist_sma20 = (row['close'] / row['1d_sma_20']) - 1
        if dist_sma20 < -0.03:
            return False, "bear_sma20"

    return True, "pass"


def backtest():
    """Run intelligent backtest on Q1 2026 data"""
    logger.info(f"\n{'='*70}")
    logger.info(f"BACKTESTING {CRYPTO} XGBOOST MODEL (Q1 2026)")
    logger.info(f"{'='*70}\n")

    # Load model
    model = joblib.load(MODEL_DIR / f'{CRYPTO.lower()}_v11_top50.joblib')
    with open(MODEL_DIR / f'{CRYPTO.lower()}_v11_features.json', 'r') as f:
        feature_cols = json.load(f)

    logger.info(f"✓ Loaded model with {len(feature_cols)} features")

    # Load test data
    df = pd.read_csv(DATA_DIR / f'{CRYPTO.lower()}_multi_tf_merged.csv')
    df['date'] = pd.to_datetime(df['date'])

    # Filter backtest period
    df_test = df[(df['date'] >= BACKTEST_START) & (df['date'] <= BACKTEST_END)].copy()
    logger.info(f"Test period: {len(df_test)} days ({BACKTEST_START} to {BACKTEST_END})")

    # Prepare features
    X_test = df_test[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)

    # Get predictions and probabilities
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    df_test['prediction'] = y_pred
    df_test['confidence'] = y_proba[:, 1]  # Confidence for class 1 (TP)

    # Simulate trading
    capital = 1000.0
    trades = []
    in_position = False
    entry_price = 0
    entry_date = None
    entry_row = None
    consecutive_losses = 0
    cooldown_until = None

    total_signals = 0
    filtered_signals = 0
    filter_reasons = {}

    for idx, row in df_test.iterrows():
        if row['prediction'] == 1 and not in_position:
            total_signals += 1

            # Cooldown check
            if cooldown_until is not None and row['date'] <= cooldown_until:
                filtered_signals += 1
                filter_reasons['cooldown'] = filter_reasons.get('cooldown', 0) + 1
                continue

            # Apply intelligent filtering
            df_idx = df_test.index.get_loc(idx)
            passes_filter, reason = intelligent_signal_filter(row, df_test, df_idx)

            if not passes_filter:
                filtered_signals += 1
                filter_reasons[reason] = filter_reasons.get(reason, 0) + 1
                continue

            # Enter position
            entry_price = row['close'] * (1 + SLIPPAGE)
            entry_date = row['date']
            entry_row = row
            in_position = True
            position_size = capital * POSITION_SIZE
            shares = position_size / entry_price
            capital -= position_size
            fees = position_size * TRADING_FEE
            capital -= fees

            logger.info(f"ENTRY: {entry_date.date()} @ ${entry_price:.2f} (Conf: {row['confidence']:.2%})")

        elif in_position:
            # Check TP/SL
            tp_price = entry_price * (1 + TP_PCT)
            sl_price = entry_price * (1 - SL_PCT)

            hit_tp = row['high'] >= tp_price
            hit_sl = row['low'] <= sl_price

            if hit_tp:
                # TP hit
                exit_price = tp_price * (1 - SLIPPAGE)
                exit_value = shares * exit_price
                fees = exit_value * TRADING_FEE
                capital += exit_value - fees

                pnl = exit_value - (shares * entry_price)
                pnl_pct = (exit_price / entry_price - 1) * 100

                trades.append({
                    'entry_date': entry_date,
                    'exit_date': row['date'],
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'exit_type': 'TP',
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'confidence': row['confidence']
                })

                logger.info(f"EXIT (TP): {row['date'].date()} @ ${exit_price:.2f} | PnL: ${pnl:.2f} ({pnl_pct:+.2f}%)")
                in_position = False
                consecutive_losses = 0

            elif hit_sl:
                # SL hit
                exit_price = sl_price * (1 - SLIPPAGE)
                exit_value = shares * exit_price
                fees = exit_value * TRADING_FEE
                capital += exit_value - fees

                pnl = exit_value - (shares * entry_price)
                pnl_pct = (exit_price / entry_price - 1) * 100

                trades.append({
                    'entry_date': entry_date,
                    'exit_date': row['date'],
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'exit_type': 'SL',
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'confidence': row['confidence']
                })

                logger.info(f"EXIT (SL): {row['date'].date()} @ ${exit_price:.2f} | PnL: ${pnl:.2f} ({pnl_pct:+.2f}%)")
                in_position = False
                consecutive_losses += 1
                if consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
                    from datetime import timedelta
                    cooldown_until = row['date'] + timedelta(days=COOLDOWN_DAYS)
                    logger.info(f"  >>> {MAX_CONSECUTIVE_LOSSES} losses -> cooldown until {cooldown_until.date()}")

    # Close any open position at last price
    if in_position:
        last_price = df_test.iloc[-1]['close']
        exit_value = shares * last_price
        capital += exit_value - (exit_value * TRADING_FEE)
        pnl = exit_value - (shares * entry_price)
        pnl_pct = (last_price / entry_price - 1) * 100
        trades.append({
            'entry_date': entry_date, 'exit_date': df_test.iloc[-1]['date'],
            'entry_price': entry_price, 'exit_price': last_price,
            'exit_type': 'EOD', 'pnl': pnl, 'pnl_pct': pnl_pct, 'confidence': 0
        })
        logger.info(f"EXIT (EOD): last day @ ${last_price:.2f} | PnL: {pnl_pct:+.2f}%")

    # Results
    trades_df = pd.DataFrame(trades)

    logger.info(f"\n{'='*70}")
    logger.info(f"BACKTEST RESULTS (Q1 2026)")
    logger.info(f"{'='*70}\n")

    logger.info(f"Signal Filtering:")
    logger.info(f"  Total signals: {total_signals}")
    logger.info(f"  Filtered out: {filtered_signals} ({filtered_signals/total_signals*100:.1f}%)")
    logger.info(f"  Trades executed: {len(trades)}")

    if filter_reasons:
        logger.info(f"\n  Filter reasons:")
        for reason, count in sorted(filter_reasons.items(), key=lambda x: -x[1]):
            logger.info(f"    {reason}: {count}")

    if len(trades) > 0:
        total_return = (capital / 1000 - 1) * 100
        wins = len(trades_df[trades_df['exit_type'] == 'TP'])
        losses = len(trades_df[trades_df['exit_type'] == 'SL'])
        win_rate = wins / len(trades) * 100

        logger.info(f"\nTrading Performance:")
        logger.info(f"  Total Trades: {len(trades)}")
        logger.info(f"  Wins: {wins} | Losses: {losses}")
        logger.info(f"  Win Rate: {win_rate:.2f}%")
        logger.info(f"  Total Return: {total_return:+.2f}%")
        logger.info(f"  Initial Capital: $1000.00")
        logger.info(f"  Final Capital: ${capital:.2f}")

        # Save results
        trades_df.to_csv(RESULTS_DIR / f'{CRYPTO.lower()}_backtest_trades.csv', index=False)

        # Save summary
        summary = {
            'crypto': CRYPTO,
            'backtest_period': f'{BACKTEST_START} to {BACKTEST_END}',
            'total_signals': total_signals,
            'filtered_signals': filtered_signals,
            'trades_executed': len(trades),
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'total_return_pct': total_return,
            'initial_capital': 1000,
            'final_capital': capital,
            'filter_reasons': filter_reasons
        }

        with open(RESULTS_DIR / f'{CRYPTO.lower()}_backtest_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"\n✓ Results saved to {RESULTS_DIR}")
    else:
        logger.info(f"\n⚠️ No trades executed (all signals filtered)")

    logger.info(f"\n{'='*70}")
    logger.info(f"BACKTEST COMPLETE")
    logger.info(f"{'='*70}")


if __name__ == "__main__":
    backtest()
