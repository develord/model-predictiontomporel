"""
Analyze if there were real trading opportunities in Q1 2026 that the model missed.
Check if TP targets (1.5%) were actually achievable.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
BACKTEST_START = '2026-01-01'
BACKTEST_END = '2026-03-24'

# Trading parameters
TP_PCT = 0.015  # 1.5%
SL_PCT = 0.0075  # 0.75%
SLIPPAGE = 0.0005

# Load data
data_file = BASE_DIR / 'data' / 'cache' / 'btc_multi_tf_merged.csv'
df = pd.read_csv(data_file)
df['date'] = pd.to_datetime(df['date'])

# Filter backtest period
df_test = df[(df['date'] >= BACKTEST_START) & (df['date'] <= BACKTEST_END)].copy()
logger.info(f"Test period: {len(df_test)} days (Q1 2026)")

# Analyze each day to see if TP was achievable
opportunities = []
total_days = 0

for idx, row in df_test.iterrows():
    total_days += 1
    date = row['date']
    entry_price = row['close'] * (1 + SLIPPAGE)
    tp_price = entry_price * (1 + TP_PCT)
    sl_price = entry_price * (1 - SL_PCT)

    # Check if TP hit before SL
    high = row['high']
    low = row['low']

    tp_hit = high >= tp_price
    sl_hit = low <= sl_price

    # Determine outcome
    if tp_hit and not sl_hit:
        outcome = 'TP_ONLY'
        opportunities.append({
            'date': date,
            'entry': entry_price,
            'high': high,
            'low': low,
            'tp_price': tp_price,
            'sl_price': sl_price,
            'outcome': outcome,
            'gain_pct': TP_PCT * 100
        })
    elif sl_hit and not tp_hit:
        outcome = 'SL_ONLY'
    elif tp_hit and sl_hit:
        outcome = 'BOTH_HIT'
        # Assume SL hit first (conservative)
    else:
        outcome = 'NEITHER'

logger.info(f"\n{'='*80}")
logger.info(f"REAL OPPORTUNITIES ANALYSIS (Q1 2026)")
logger.info(f"{'='*80}")
logger.info(f"Total days: {total_days}")
logger.info(f"TP achievable days: {len(opportunities)} ({len(opportunities)/total_days*100:.1f}%)")

if opportunities:
    logger.info(f"\nBest opportunities (TP achievable):")
    for opp in opportunities[:15]:  # Show first 15
        logger.info(f"  {opp['date'].date()}: Entry ${opp['entry']:.2f} -> High ${opp['high']:.2f} (TP @ ${opp['tp_price']:.2f}) ✓")

    # Calculate what perfect strategy would achieve
    perfect_return = len(opportunities) * (TP_PCT - 0.001 - SLIPPAGE)  # Account for fees
    logger.info(f"\nPERFECT STRATEGY (if we knew which days):")
    logger.info(f"  Trades: {len(opportunities)}")
    logger.info(f"  Win Rate: 100%")
    logger.info(f"  Total Return: {perfect_return*100:.2f}%")
    logger.info(f"  Initial: $1000 -> Final: ${1000*(1+perfect_return):.2f}")
else:
    logger.info(f"\n⚠️ NO TP-only opportunities found in Q1 2026!")
    logger.info(f"This means the market was too choppy for 1.5% TP strategy")

# Also check price movements
logger.info(f"\n{'='*80}")
logger.info(f"PRICE MOVEMENT ANALYSIS")
logger.info(f"{'='*80}")
start_price = df_test.iloc[0]['close']
end_price = df_test.iloc[-1]['close']
max_price = df_test['high'].max()
min_price = df_test['low'].min()

logger.info(f"Start: ${start_price:.2f} (2026-01-01)")
logger.info(f"End: ${end_price:.2f} (2026-03-24)")
logger.info(f"Change: {(end_price/start_price - 1)*100:+.2f}%")
logger.info(f"Max: ${max_price:.2f}")
logger.info(f"Min: ${min_price:.2f}")
logger.info(f"Range: {(max_price/min_price - 1)*100:.1f}%")

# Check daily volatility
df_test['daily_change'] = df_test['close'].pct_change() * 100
avg_daily_move = df_test['daily_change'].abs().mean()
logger.info(f"Average daily move: {avg_daily_move:.2f}%")
logger.info(f"TP target: {TP_PCT*100:.2f}%")

if avg_daily_move < TP_PCT * 100:
    logger.info(f"\n⚠️ Average daily move ({avg_daily_move:.2f}%) < TP target ({TP_PCT*100:.2f}%)")
    logger.info(f"Market not volatile enough for 1.5% TP strategy in Q1 2026!")
