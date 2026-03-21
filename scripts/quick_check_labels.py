"""Quick label distribution check without Unicode issues"""
import pandas as pd
from pathlib import Path

cache_dir = Path(__file__).parent.parent / 'data' / 'cache'

for crypto in ['btc', 'eth', 'sol']:
    csv_file = cache_dir / f'{crypto}_multi_tf_merged.csv'

    if not csv_file.exists():
        print(f'\n{crypto.upper()}: FILE NOT FOUND')
        continue

    df = pd.read_csv(csv_file, index_col=0)

    if 'triple_barrier_label' not in df.columns:
        print(f'\n{crypto.upper()}: NO TRIPLE_BARRIER_LABEL COLUMN')
        continue

    dist = df['triple_barrier_label'].value_counts().sort_index()
    dist_pct = df['triple_barrier_label'].value_counts(normalize=True).sort_index() * 100

    print(f'\n{crypto.upper()}:')
    print(f'  SL (-1):     {dist.get(-1.0, 0):5d} ({dist_pct.get(-1.0, 0):5.1f}%)')
    print(f'  Timeout (0): {dist.get(0.0, 0):5d} ({dist_pct.get(0.0, 0):5.1f}%)')
    print(f'  TP (+1):     {dist.get(1.0, 0):5d} ({dist_pct.get(1.0, 0):5.1f}%)')
    print(f'  Total:       {df["triple_barrier_label"].notna().sum():5d}')

    # Check for issues
    if dist.get(0.0, 0) == 0:
        print(f'  WARNING: NO TIMEOUT LABELS!')
