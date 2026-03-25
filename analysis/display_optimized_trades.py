"""
Display All Optimized Trades - Phase 1 Quick Wins
==================================================
Show all 91 trades from optimized backtest one by one
"""

import sys
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent))


def run_and_display_optimized_trades():
    """Run optimized backtest and display all trades"""

    models = [
        ('btc', 'baseline', 0.55),
        ('eth', 'top50', 0.50),
        ('sol', 'optuna', 0.65)
    ]

    print("=" * 160)
    print("ALL OPTIMIZED TRADES - PHASE 1 QUICK WINS (V11 OPTIMIZED)")
    print("=" * 160)
    print()
    print("Improvements applied:")
    print("  - QW #1: Adaptive thresholds (BTC=0.55, ETH=0.50, SOL=0.65)")
    print("  - QW #2: Filter low confidence zone (50-53%)")
    print("  - QW #5: Market regime filter (SMA50 > SMA200)")
    print("  - QW #10: Consecutive loss protection (pause after 3 losses)")
    print("  - QW #11: Day-of-week filter (avoid Friday)")
    print()
    print("=" * 160)
    print()

    all_trades = []
    trade_counter = 0

    for crypto, mode, threshold in models:
        print(f"\n{'='*160}")
        print(f"{crypto.upper()} - {mode.upper()} (Threshold: {threshold})")
        print('='*160)
        print()

        # Load model
        model_file = Path(__file__).parent.parent / 'models' / f'{crypto}_v11_{mode}.joblib'
        model = joblib.load(model_file)

        # Load stats
        stats_file = Path(__file__).parent.parent / 'models' / f'{crypto}_v11_{mode}_stats.json'
        with open(stats_file, 'r') as f:
            stats = json.load(f)

        # Load data
        cache_file = Path(__file__).parent.parent / 'data' / 'cache' / f'{crypto}_multi_tf_merged.csv'
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)

        # Prepare features
        exclude_cols = [
            'open', 'high', 'low', 'close', 'volume',
            'label_class', 'label_numeric', 'price_target_pct',
            'future_price', 'triple_barrier_label'
        ]

        all_features = [col for col in df.columns if col not in exclude_cols]

        if mode == 'top50':
            top_features = [f['feature'] for f in stats['top_features'][:50]]
            feature_cols = [f for f in top_features if f in df.columns]
        else:
            feature_cols = all_features

        # Filter to test period
        df_test = df[df.index >= '2025-01-01'].copy()

        # Prepare features
        X_test = df_test[feature_cols].fillna(0).values
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

        # Predict
        prob_tp = model.predict_proba(X_test)[:, 1]
        df_test['prob_tp'] = prob_tp

        # Get day of week
        df_test['day_of_week'] = df_test.index.dayofweek

        # Market regime filter
        has_sma_filter = '1d_sma_50' in df_test.columns and '1d_sma_200' in df_test.columns
        if has_sma_filter:
            df_test['bullish_regime'] = df_test['1d_sma_50'] > df_test['1d_sma_200']
        else:
            df_test['bullish_regime'] = True

        # Signal generation with all filters
        df_test['signal'] = (
            (prob_tp > threshold) &
            ((prob_tp < 0.50) | (prob_tp >= 0.53)) &  # QW #2
            (df_test['day_of_week'] != 4) &  # QW #11
            (df_test['bullish_regime'])  # QW #5
        )

        # Simulate trades
        trades = []
        position = None
        consecutive_losses = 0
        tp_pct = 1.5
        sl_pct = 0.75

        for i in range(len(df_test)):
            row = df_test.iloc[i]

            # Handle existing position
            if position is not None:
                high = df_test['high'].iloc[i]
                low = df_test['low'].iloc[i]

                entry_price = position['entry_price']
                tp_price = entry_price * (1 + tp_pct / 100)
                sl_price = entry_price * (1 - sl_pct / 100)

                hit_tp = high >= tp_price
                hit_sl = low <= sl_price

                if hit_tp or hit_sl:
                    exit_price = tp_price if hit_tp else sl_price
                    exit_type = 'TP' if hit_tp else 'SL'
                    pnl_pct = ((exit_price - entry_price) / entry_price) * 100

                    position['exit_time'] = row.name
                    position['exit_price'] = exit_price
                    position['exit_type'] = exit_type
                    position['pnl_pct'] = pnl_pct
                    position['bars_held'] = i - position['entry_idx']
                    position['result'] = 'WIN' if hit_tp else 'LOSS'

                    trades.append(position)
                    position = None

                    # Track consecutive losses
                    if hit_tp:
                        consecutive_losses = 0
                    else:
                        consecutive_losses += 1

            # QW #10: Skip after 3 consecutive losses
            elif row['signal'] and consecutive_losses < 3:
                position = {
                    'crypto': crypto.upper(),
                    'trade_id': len(trades) + 1,
                    'entry_time': row.name,
                    'entry_idx': i,
                    'entry_price': row['close'],
                    'prob_tp': row['prob_tp'],
                    'day_of_week': row['day_of_week']
                }

        # Display trades
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

        for trade in trades:
            trade_counter += 1

            print(f"Trade #{trade_counter} - {trade['crypto']} Trade #{trade['trade_id']}")
            print("-" * 160)
            print(f"  Entry Time:    {trade['entry_time']}")
            print(f"  Entry Price:   ${trade['entry_price']:,.2f}")
            print(f"  Exit Time:     {trade['exit_time']}")
            print(f"  Exit Price:    ${trade['exit_price']:,.2f}")
            print(f"  Exit Type:     {trade['exit_type']}")
            print(f"  PnL:           {trade['pnl_pct']:+.2f}%")
            print(f"  Prob TP:       {trade['prob_tp']*100:.2f}%")
            print(f"  Bars Held:     {trade['bars_held']}")
            print(f"  Day of Week:   {day_names[int(trade['day_of_week'])]}")

            result_emoji = "WIN" if trade['result'] == 'WIN' else "LOSS"
            print(f"  Result:        {result_emoji}")
            print()

            all_trades.append(trade)

    # Final summary
    print("\n" + "=" * 160)
    print("FINAL SUMMARY - ALL OPTIMIZED TRADES")
    print("=" * 160)
    print()

    trades_df = pd.DataFrame(all_trades)

    print(f"Total Trades: {len(trades_df)}")
    print()

    # Summary by crypto
    print(f"{'Crypto':<8} {'Trades':<8} {'Wins':<6} {'Losses':<8} {'Win Rate':<10} {'Total PnL':<12} {'Avg PnL':<10}")
    print("-" * 160)

    for crypto in ['BTC', 'ETH', 'SOL']:
        crypto_trades = trades_df[trades_df['crypto'] == crypto]

        if len(crypto_trades) == 0:
            continue

        total = len(crypto_trades)
        wins = len(crypto_trades[crypto_trades['result'] == 'WIN'])
        losses = len(crypto_trades[crypto_trades['result'] == 'LOSS'])
        win_rate = (wins / total) * 100
        total_pnl = crypto_trades['pnl_pct'].sum()
        avg_pnl = crypto_trades['pnl_pct'].mean()

        print(f"{crypto:<8} {total:<8} {wins:<6} {losses:<8} {win_rate:<10.2f} {total_pnl:+<12.2f}% {avg_pnl:+<10.2f}%")

    # Overall
    print("-" * 160)
    total_all = len(trades_df)
    wins_all = len(trades_df[trades_df['result'] == 'WIN'])
    losses_all = len(trades_df[trades_df['result'] == 'LOSS'])
    wr_all = (wins_all / total_all) * 100
    total_pnl_all = trades_df['pnl_pct'].sum()
    avg_pnl_all = trades_df['pnl_pct'].mean()

    print(f"{'TOTAL':<8} {total_all:<8} {wins_all:<6} {losses_all:<8} {wr_all:<10.2f} {total_pnl_all:+<12.2f}% {avg_pnl_all:+<10.2f}%")

    print()
    print("=" * 160)
    print()


if __name__ == '__main__':
    run_and_display_optimized_trades()
