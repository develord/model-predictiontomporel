"""
V12 Dynamic Labels - ATR-based Triple Barrier Labeling
======================================================
Instead of fixed TP/SL (1.5% / 0.75%), TP and SL adapt to current volatility
using ATR (Average True Range).

Key Innovation:
- High volatility period: wider TP/SL -> fewer premature stops
- Low volatility period: tighter TP/SL -> faster trades, less exposure
- R:R ratio maintained >= 1.5 at all times

Compared to V11:
- V11: Fixed TP=1.5%, SL=0.75% for ALL market conditions
- V12: TP=2.0*ATR%, SL=1.0*ATR% adapting to each candle's volatility
"""

import sys
import numpy as np
import pandas as pd
from typing import Dict, Tuple
import json
from pathlib import Path

# Ensure project root is in path for importing V11 modules
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def calculate_atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate ATR as percentage of close price.

    Returns:
        Series of ATR values in percentage terms (e.g., 2.5 means 2.5%)
    """
    high = df['high']
    low = df['low']
    close = df['close']

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()

    # Convert to percentage of close price
    atr_pct = (atr / close) * 100

    return atr_pct


def apply_dynamic_triple_barrier(
    df: pd.DataFrame,
    tp_atr_mult: float = 2.0,
    sl_atr_mult: float = 1.0,
    atr_period: int = 14,
    lookahead_candles: int = 7,
    min_tp_pct: float = 0.5,
    max_tp_pct: float = 5.0,
    min_sl_pct: float = 0.25,
    max_sl_pct: float = 2.5,
    min_rr_ratio: float = 1.5
) -> pd.DataFrame:
    """
    Apply Triple Barrier Labeling with dynamic ATR-based TP/SL.

    Each candle gets its own TP/SL based on current ATR:
      TP = ATR_pct * tp_atr_mult (clamped to [min_tp, max_tp])
      SL = ATR_pct * sl_atr_mult (clamped to [min_sl, max_sl])

    Args:
        df: DataFrame with OHLCV data
        tp_atr_mult: ATR multiplier for Take Profit
        sl_atr_mult: ATR multiplier for Stop Loss
        atr_period: ATR calculation period
        lookahead_candles: Max candles to wait for TP/SL
        min_tp_pct / max_tp_pct: TP percentage bounds
        min_sl_pct / max_sl_pct: SL percentage bounds
        min_rr_ratio: Minimum risk/reward ratio enforced

    Returns:
        DataFrame with columns:
          - triple_barrier_label: +1 (TP), -1 (SL), 0 (timeout)
          - dynamic_tp_pct: actual TP% used for this candle
          - dynamic_sl_pct: actual SL% used for this candle
          - atr_pct: ATR percentage at this candle
    """
    result = df.copy()

    # Calculate ATR percentage
    atr_pct = calculate_atr_series(df, atr_period)
    result['atr_pct_14'] = atr_pct

    labels = []
    tp_pcts = []
    sl_pcts = []

    for i in range(len(df)):
        # Not enough history for ATR or not enough future data
        if pd.isna(atr_pct.iloc[i]) or i + lookahead_candles >= len(df):
            labels.append(np.nan)
            tp_pcts.append(np.nan)
            sl_pcts.append(np.nan)
            continue

        current_atr = atr_pct.iloc[i]

        # Calculate dynamic TP/SL from ATR
        tp = current_atr * tp_atr_mult
        sl = current_atr * sl_atr_mult

        # Clamp to bounds
        tp = np.clip(tp, min_tp_pct, max_tp_pct)
        sl = np.clip(sl, min_sl_pct, max_sl_pct)

        # Enforce minimum R:R ratio
        if tp / sl < min_rr_ratio:
            # Tighten SL to enforce R:R
            sl = tp / min_rr_ratio
            sl = max(sl, min_sl_pct)  # Don't go below min SL

        tp_pcts.append(tp)
        sl_pcts.append(sl)

        entry_price = df['close'].iloc[i]
        tp_price = entry_price * (1 + tp / 100)
        sl_price = entry_price * (1 - sl / 100)

        # Check future candles
        label = 0  # Default: timeout
        for j in range(1, lookahead_candles + 1):
            if i + j >= len(df):
                break

            future_high = df['high'].iloc[i + j]
            future_low = df['low'].iloc[i + j]

            # Check both barriers on same candle - use which is closer to open
            tp_hit = future_high >= tp_price
            sl_hit = future_low <= sl_price

            if tp_hit and sl_hit:
                # Both hit on same candle - use open to determine which hit first
                future_open = df['open'].iloc[i + j]
                # If open is closer to SL, SL likely hit first
                if abs(future_open - sl_price) < abs(future_open - tp_price):
                    label = -1
                else:
                    label = 1
                break
            elif tp_hit:
                label = 1
                break
            elif sl_hit:
                label = -1
                break

        labels.append(label)

    result['triple_barrier_label'] = labels
    result['dynamic_tp_pct'] = tp_pcts
    result['dynamic_sl_pct'] = sl_pcts

    return result


def generate_v12_labels(
    df: pd.DataFrame,
    timeframe: str,
    crypto: str = 'btc',
    config: dict = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Generate V12 dynamic labels for a given timeframe and crypto.

    Args:
        df: DataFrame with OHLCV data
        timeframe: '4h', '1d', or '1w'
        crypto: 'btc', 'eth', or 'sol'
        config: V12 config dict (loads from file if None)

    Returns:
        Tuple of (DataFrame with labels, statistics dict)
    """
    # Load config
    if config is None:
        config_path = Path(__file__).parent.parent / 'config' / 'v12_config.json'
        with open(config_path, 'r') as f:
            config = json.load(f)

    # Get crypto-specific multipliers
    crypto_config = config.get(crypto.upper(), {})
    global_config = config.get('dynamic_tp_sl', {})

    tp_mult = crypto_config.get('tp_atr_multiplier', global_config.get('tp_atr_multiplier', 2.0))
    sl_mult = crypto_config.get('sl_atr_multiplier', global_config.get('sl_atr_multiplier', 1.0))
    atr_period = global_config.get('atr_period', 14)

    # Lookahead by timeframe
    lookahead_config = {
        '4h': 42,   # 7 days
        '1d': 7,    # 7 days
        '1w': 1     # 1 week
    }
    lookahead = lookahead_config.get(timeframe, 7)

    # Apply dynamic triple barrier
    result = apply_dynamic_triple_barrier(
        df,
        tp_atr_mult=tp_mult,
        sl_atr_mult=sl_mult,
        atr_period=atr_period,
        lookahead_candles=lookahead,
        min_tp_pct=global_config.get('min_tp_pct', 0.5),
        max_tp_pct=global_config.get('max_tp_pct', 5.0),
        min_sl_pct=global_config.get('min_sl_pct', 0.25),
        max_sl_pct=global_config.get('max_sl_pct', 2.5),
        min_rr_ratio=global_config.get('risk_reward_min', 1.5)
    )

    # Also generate V11-style fixed labels for comparison
    from features.labels import apply_triple_barrier as apply_fixed_barrier
    fixed_result = apply_fixed_barrier(df, tp_pct=1.5, sl_pct=0.75, lookahead_candles=lookahead)
    result['v11_fixed_label'] = fixed_result['triple_barrier_label']

    # Statistics
    valid = result[result['triple_barrier_label'].notna()]
    n_tp = int((valid['triple_barrier_label'] == 1).sum())
    n_sl = int((valid['triple_barrier_label'] == -1).sum())
    n_timeout = int((valid['triple_barrier_label'] == 0).sum())
    total = len(valid)

    # ATR stats
    atr_valid = result['atr_pct_14'].dropna()
    tp_valid = result['dynamic_tp_pct'].dropna()
    sl_valid = result['dynamic_sl_pct'].dropna()

    # Compare with V11 fixed labels
    v11_valid = result[result['v11_fixed_label'].notna()]
    v11_tp = int((v11_valid['v11_fixed_label'] == 1).sum())
    v11_sl = int((v11_valid['v11_fixed_label'] == -1).sum())

    stats = {
        'crypto': crypto,
        'timeframe': timeframe,
        'version': 'v12_dynamic_atr',
        'tp_atr_multiplier': tp_mult,
        'sl_atr_multiplier': sl_mult,
        'atr_period': atr_period,
        'total_samples': total,
        'label_distribution': {
            'TP': n_tp,
            'SL': n_sl,
            'Timeout': n_timeout,
            'TP_pct': round(n_tp / total * 100, 2) if total > 0 else 0,
            'SL_pct': round(n_sl / total * 100, 2) if total > 0 else 0,
            'Timeout_pct': round(n_timeout / total * 100, 2) if total > 0 else 0
        },
        'atr_stats': {
            'mean': round(float(atr_valid.mean()), 4),
            'std': round(float(atr_valid.std()), 4),
            'min': round(float(atr_valid.min()), 4),
            'max': round(float(atr_valid.max()), 4),
            'median': round(float(atr_valid.median()), 4)
        },
        'dynamic_tp_stats': {
            'mean': round(float(tp_valid.mean()), 4),
            'std': round(float(tp_valid.std()), 4),
            'min': round(float(tp_valid.min()), 4),
            'max': round(float(tp_valid.max()), 4)
        },
        'dynamic_sl_stats': {
            'mean': round(float(sl_valid.mean()), 4),
            'std': round(float(sl_valid.std()), 4),
            'min': round(float(sl_valid.min()), 4),
            'max': round(float(sl_valid.max()), 4)
        },
        'vs_v11_fixed': {
            'v11_tp': v11_tp,
            'v11_sl': v11_sl,
            'v12_tp': n_tp,
            'v12_sl': n_sl,
            'tp_change': n_tp - v11_tp,
            'sl_change': n_sl - v11_sl
        }
    }

    return result, stats


def print_label_comparison(stats: Dict):
    """Pretty print V12 vs V11 label comparison"""
    crypto = stats['crypto'].upper()

    print(f"\n{'='*60}")
    print(f"  {crypto} - V12 Dynamic ATR Labels")
    print(f"{'='*60}")

    dist = stats['label_distribution']
    print(f"  TP: {dist['TP']:5d} ({dist['TP_pct']:5.1f}%)")
    print(f"  SL: {dist['SL']:5d} ({dist['SL_pct']:5.1f}%)")
    print(f"  TO: {dist['Timeout']:5d} ({dist['Timeout_pct']:5.1f}%)")

    atr = stats['atr_stats']
    print(f"\n  ATR(14) %: mean={atr['mean']:.2f}%, median={atr['median']:.2f}%, range=[{atr['min']:.2f}%, {atr['max']:.2f}%]")

    tp_s = stats['dynamic_tp_stats']
    sl_s = stats['dynamic_sl_stats']
    print(f"  Dynamic TP: mean={tp_s['mean']:.2f}%, range=[{tp_s['min']:.2f}%, {tp_s['max']:.2f}%]")
    print(f"  Dynamic SL: mean={sl_s['mean']:.2f}%, range=[{sl_s['min']:.2f}%, {sl_s['max']:.2f}%]")
    print(f"  Avg R:R = {tp_s['mean']/sl_s['mean']:.2f}:1")

    vs = stats['vs_v11_fixed']
    print(f"\n  vs V11 (fixed 1.5%/0.75%):")
    print(f"    V11: TP={vs['v11_tp']}, SL={vs['v11_sl']}")
    print(f"    V12: TP={vs['v12_tp']}, SL={vs['v12_sl']}")
    print(f"    Delta: TP {vs['tp_change']:+d}, SL {vs['sl_change']:+d}")
