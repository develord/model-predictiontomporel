import pandas as pd

# Load BTC trades
df = pd.read_csv(r'C:\Users\moham\Desktop\crypto\crypto_v10_multi_tf\results\backtest_2026_btc_trades.csv')

print('BTC TRADES ANALYSIS')
print('='*60)
print(f'Total trades: {len(df)}')
print(f'Wins: {len(df[df["result"] == "WIN"])} ({len(df[df["result"] == "WIN"])/len(df)*100:.1f}%)')
print(f'Losses: {len(df[df["result"] == "LOSE"])} ({len(df[df["result"] == "LOSE"])/len(df)*100:.1f}%)')

print(f'\nConfidence distribution:')
print(f'  Min: {df["prob_tp"].min():.1%}')
print(f'  Max: {df["prob_tp"].max():.1%}')
print(f'  Mean: {df["prob_tp"].mean():.1%}')
print(f'  Median: {df["prob_tp"].median():.1%}')

wins = df[df["result"] == "WIN"]
losses = df[df["result"] == "LOSE"]

print(f'\nWins avg confidence: {wins["prob_tp"].mean():.1%}')
print(f'Losses avg confidence: {losses["prob_tp"].mean():.1%}')

print(f'\nTRADES BY CONFIDENCE THRESHOLD:')
print('='*60)
for thresh in [0.60, 0.65, 0.70, 0.75, 0.80]:
    trades_above = df[df['prob_tp'] >= thresh]
    if len(trades_above) > 0:
        wins_count = len(trades_above[trades_above['result'] == 'WIN'])
        wr = wins_count / len(trades_above) * 100
        total_pnl = trades_above['pnl'].sum()
        roi = (total_pnl / 10000) * 100
        print(f'>={thresh:.0%}: {len(trades_above):2d} trades, WR={wr:5.1f}%, ROI={roi:+6.2f}%')
    else:
        print(f'>={thresh:.0%}: 0 trades')
