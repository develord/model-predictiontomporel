"""
V12 Multi-Horizon Labels
=========================
Generate triple barrier labels for 3 horizons:
- 3 days (short-term)
- 5 days (medium-term)
- 7 days (standard, same as V11)

Each horizon uses the SAME fixed TP/SL (1.5%/0.75%) from V11 labels.
The difference is only the lookahead window.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def apply_triple_barrier(df, tp_pct=1.5, sl_pct=0.75, lookahead=7):
    """Triple barrier with configurable lookahead."""
    labels = []
    for i in range(len(df)):
        if i + lookahead >= len(df):
            labels.append(np.nan)
            continue

        entry = df['close'].iloc[i]
        tp_price = entry * (1 + tp_pct / 100)
        sl_price = entry * (1 - sl_pct / 100)

        label = 0
        for j in range(1, lookahead + 1):
            if i + j >= len(df):
                break
            h = df['high'].iloc[i + j]
            l = df['low'].iloc[i + j]

            if h >= tp_price:
                label = 1
                break
            elif l <= sl_price:
                label = -1
                break

        labels.append(label)

    return labels


def generate_multi_horizon_labels(df):
    """
    Generate labels for 3 horizons on 1d data.
    Returns df with columns: label_3d, label_5d, label_7d
    """
    result = df.copy()

    for horizon in [3, 5, 7]:
        col = f'label_{horizon}d'
        result[col] = apply_triple_barrier(df, tp_pct=1.5, sl_pct=0.75, lookahead=horizon)

    return result
