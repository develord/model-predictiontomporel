"""Quick analysis of LONG trade failures"""
import pandas as pd
import numpy as np

df = pd.read_csv('results/btc_backtest_cnn-only.csv')
df['date'] = pd.to_datetime(df['date'])
lt = df[df['dir']=='LONG']

print('=== ALL LONG TRADES ===')
for _, r in lt.iterrows():
    win = 'WIN' if r['exit_type']=='TP' else ('LOSE' if r['exit_type']=='SL' else 'EOD')
    print(f"  {r['date'].strftime('%Y-%m-%d')} | CNN:{r['cnn_conf']:.0%} | {r['exit_type']} | PnL:{r['pnl']:+.2f}% | {win}")

tpsl = lt[lt['exit_type'].isin(['TP','SL'])]
wins = tpsl[tpsl['exit_type']=='TP']
loses = tpsl[tpsl['exit_type']=='SL']
print(f'\n=== TP/SL STATS ===')
print(f'  Wins: {len(wins)}, Loses: {len(loses)}, WR: {len(wins)/len(tpsl)*100:.0f}%')
print(f'  Avg win PnL: {wins["pnl"].mean():+.2f}%')
print(f'  Avg lose PnL: {loses["pnl"].mean():+.2f}%')

print(f'\n=== WR BY CONFIDENCE ===')
for lo, hi in [(0.55,0.65),(0.65,0.75),(0.75,0.85),(0.85,1.0)]:
    m = tpsl[(tpsl['cnn_conf']>=lo)&(tpsl['cnn_conf']<hi)]
    if len(m)>0:
        wr = (m['exit_type']=='TP').mean()*100
        print(f'  CNN {lo:.0%}-{hi:.0%}: {len(m)} trades, WR: {wr:.0f}%')

print(f'\n=== WR BY MONTH ===')
for mo in sorted(tpsl['date'].dt.month.unique()):
    m = tpsl[tpsl['date'].dt.month==mo]
    if len(m)>0:
        wr = (m['exit_type']=='TP').mean()*100
        print(f'  Month {mo}: {len(m)} trades, WR: {wr:.0f}%')

# Load features to understand market state during losses
features = pd.read_csv('data/cache/btc_features.csv')
features['date'] = pd.to_datetime(features['date'])

print(f'\n=== MARKET CONTEXT DURING LONG SL ===')
for _, r in loses.iterrows():
    feat = features[features['date']==r['date']]
    if len(feat)==0:
        continue
    f = feat.iloc[0]
    regime = 'BULL' if f.get('regime_bull',0)==1 else ('BEAR' if f.get('regime_bear',0)==1 else 'RANGE')
    rsi = f.get('1d_rsi_14', 0)
    trend = f.get('trend_score', 0)
    sma_dist = f.get('distance_from_sma50', 0)
    vol_reg = f.get('volatility_regime', 0)
    print(f"  {r['date'].strftime('%Y-%m-%d')} | {regime} | RSI:{rsi:.0f} | Trend:{trend:.1f} | SMA50dist:{sma_dist:.3f} | VolReg:{vol_reg:.2f}")

print(f'\n=== MARKET CONTEXT DURING LONG TP ===')
for _, r in wins.iterrows():
    feat = features[features['date']==r['date']]
    if len(feat)==0:
        continue
    f = feat.iloc[0]
    regime = 'BULL' if f.get('regime_bull',0)==1 else ('BEAR' if f.get('regime_bear',0)==1 else 'RANGE')
    rsi = f.get('1d_rsi_14', 0)
    trend = f.get('trend_score', 0)
    sma_dist = f.get('distance_from_sma50', 0)
    vol_reg = f.get('volatility_regime', 0)
    print(f"  {r['date'].strftime('%Y-%m-%d')} | {regime} | RSI:{rsi:.0f} | Trend:{trend:.1f} | SMA50dist:{sma_dist:.3f} | VolReg:{vol_reg:.2f}")
