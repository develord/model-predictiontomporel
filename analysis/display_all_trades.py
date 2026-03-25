"""
Display All Trades - BTC, ETH, SOL
===================================
Show all trades from all 3 cryptos in comprehensive table
"""

import pandas as pd
from pathlib import Path

def display_all_trades():
    """Display all trades from all cryptos"""

    results_dir = Path(__file__).parent.parent / 'backtesting' / 'results'

    models = [
        ('BTC', 'baseline', 'btc_baseline_detailed_trades.csv'),
        ('ETH', 'top50', 'eth_top50_detailed_trades.csv'),
        ('SOL', 'optuna', 'sol_optuna_detailed_trades.csv')
    ]

    print("=" * 160)
    print("ALL TRADES - BTC, ETH, SOL (V11 BASELINE RESULTS)")
    print("=" * 160)
    print()

    all_trades = []

    for crypto, mode, filename in models:
        csv_file = results_dir / filename

        if not csv_file.exists():
            print(f"WARNING: {filename} not found")
            continue

        df = pd.read_csv(csv_file)
        df['crypto'] = crypto
        all_trades.append(df)

    # Combine all trades
    combined_df = pd.concat(all_trades, ignore_index=True)

    # Sort by entry time
    combined_df['entry_time'] = pd.to_datetime(combined_df['entry_time'])
    combined_df = combined_df.sort_values('entry_time')

    print(f"Total Trades: {len(combined_df)}")
    print()

    # Display table
    print(f"{'ID':<5} {'Crypto':<6} {'Entry Time':<20} {'Entry $':<12} {'Exit $':<12} "
          f"{'Exit':<6} {'PnL':<8} {'Prob TP':<9} {'Prob SL':<9} {'Conf':<7} "
          f"{'Bars':<5} {'Result':<6}")
    print("-" * 160)

    for idx, row in combined_df.iterrows():
        trade_id = row['trade_id']
        crypto = row['crypto']
        entry_time = row['entry_time'].strftime('%Y-%m-%d %H:%M')
        entry_price = f"${float(row['entry_price'].replace('$', '').replace(',', '')):,.2f}"
        exit_price = f"${float(row['exit_price'].replace('$', '').replace(',', '')):,.2f}"
        exit_type = row['exit_type']
        pnl = row['pnl_pct']
        prob_tp = row['prob_tp']
        prob_sl = row['prob_sl']
        confidence = row['confidence']
        bars = row['bars_held']
        result = row['result']

        # Color coding for result
        result_display = result

        print(f"{trade_id:<5} {crypto:<6} {entry_time:<20} {entry_price:<12} {exit_price:<12} "
              f"{exit_type:<6} {pnl:<8} {prob_tp:<9} {prob_sl:<9} {confidence:<7} "
              f"{bars:<5} {result_display:<6}")

    print("-" * 160)
    print()

    # Summary by crypto
    print("=" * 160)
    print("SUMMARY BY CRYPTO")
    print("=" * 160)
    print()

    print(f"{'Crypto':<8} {'Total':<7} {'Wins':<6} {'Losses':<8} {'Win Rate':<10} "
          f"{'Total PnL':<12} {'Avg PnL':<10} {'Best':<8} {'Worst':<8}")
    print("-" * 160)

    for crypto, mode, _ in models:
        crypto_trades = combined_df[combined_df['crypto'] == crypto]

        if len(crypto_trades) == 0:
            continue

        total = len(crypto_trades)
        wins = len(crypto_trades[crypto_trades['result'] == 'WIN'])
        losses = len(crypto_trades[crypto_trades['result'] == 'LOSS'])
        win_rate = (wins / total) * 100

        # Extract numeric PnL
        pnl_values = crypto_trades['pnl_pct'].str.rstrip('%').astype(float)
        total_pnl = pnl_values.sum()
        avg_pnl = pnl_values.mean()
        best_pnl = pnl_values.max()
        worst_pnl = pnl_values.min()

        print(f"{crypto:<8} {total:<7} {wins:<6} {losses:<8} {win_rate:<10.2f} "
              f"{total_pnl:+<12.2f}% {avg_pnl:+<10.2f}% {best_pnl:+<8.2f}% {worst_pnl:+<8.2f}%")

    # Overall summary
    print("-" * 160)
    total_all = len(combined_df)
    wins_all = len(combined_df[combined_df['result'] == 'WIN'])
    losses_all = len(combined_df[combined_df['result'] == 'LOSS'])
    wr_all = (wins_all / total_all) * 100

    all_pnl = combined_df['pnl_pct'].str.rstrip('%').astype(float)
    total_pnl_all = all_pnl.sum()
    avg_pnl_all = all_pnl.mean()
    best_pnl_all = all_pnl.max()
    worst_pnl_all = all_pnl.min()

    print(f"{'TOTAL':<8} {total_all:<7} {wins_all:<6} {losses_all:<8} {wr_all:<10.2f} "
          f"{total_pnl_all:+<12.2f}% {avg_pnl_all:+<10.2f}% {best_pnl_all:+<8.2f}% {worst_pnl_all:+<8.2f}%")

    print()
    print("=" * 160)
    print()

    # Breakdown by result type
    print("BREAKDOWN BY EXIT TYPE")
    print("=" * 160)
    print()

    print(f"{'Crypto':<8} {'TP Trades':<10} {'TP Avg':<10} {'SL Trades':<10} {'SL Avg':<10} "
          f"{'Avg Bars TP':<12} {'Avg Bars SL':<12}")
    print("-" * 160)

    for crypto, mode, _ in models:
        crypto_trades = combined_df[combined_df['crypto'] == crypto]

        if len(crypto_trades) == 0:
            continue

        tp_trades = crypto_trades[crypto_trades['exit_type'] == 'TP']
        sl_trades = crypto_trades[crypto_trades['exit_type'] == 'SL']

        tp_count = len(tp_trades)
        sl_count = len(sl_trades)

        tp_avg = tp_trades['pnl_pct'].str.rstrip('%').astype(float).mean() if tp_count > 0 else 0
        sl_avg = sl_trades['pnl_pct'].str.rstrip('%').astype(float).mean() if sl_count > 0 else 0

        tp_bars = tp_trades['bars_held'].mean() if tp_count > 0 else 0
        sl_bars = sl_trades['bars_held'].mean() if sl_count > 0 else 0

        print(f"{crypto:<8} {tp_count:<10} {tp_avg:+<10.2f}% {sl_count:<10} {sl_avg:+<10.2f}% "
              f"{tp_bars:<12.2f} {sl_bars:<12.2f}")

    print()
    print("=" * 160)
    print()


if __name__ == '__main__':
    display_all_trades()
