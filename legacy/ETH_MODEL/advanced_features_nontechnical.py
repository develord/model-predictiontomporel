"""
ADVANCED NON-TECHNICAL FEATURES
================================

Features non-techniques avancées pour améliorer la prédiction ETH:

1. Market Structure Features (détection de patterns)
2. Volatility Regime Detection (clustering)
3. Volume Profile Analysis
4. Price Action Patterns (candlestick patterns)
5. Market Microstructure (order flow estimation)
6. Trend Strength Indicators
7. Support/Resistance Levels
8. Market Cycle Detection
9. Momentum Shifts Detection
10. Volatility Breakout Detection

Date: 2026-03-29
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.signal import find_peaks
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def detect_candlestick_patterns(df):
    """
    Détecte les patterns de chandelier classiques.

    Returns patterns binaires (0/1) pour chaque type.
    """
    patterns = {}

    # Body and shadow calculations
    body = abs(df['close'] - df['open'])
    upper_shadow = df['high'] - df[['close', 'open']].max(axis=1)
    lower_shadow = df[['close', 'open']].min(axis=1) - df['low']
    total_range = df['high'] - df['low']

    # Bullish/Bearish
    is_bullish = (df['close'] > df['open']).astype(int)
    is_bearish = (df['close'] < df['open']).astype(int)

    # 1. DOJI (indecision)
    patterns['doji'] = ((body / (total_range + 1e-10)) < 0.1).astype(int)

    # 2. HAMMER (bullish reversal)
    # Small body at top, long lower shadow
    patterns['hammer'] = (
        (lower_shadow > 2 * body) &
        (upper_shadow < body) &
        is_bullish
    ).astype(int)

    # 3. SHOOTING STAR (bearish reversal)
    # Small body at bottom, long upper shadow
    patterns['shooting_star'] = (
        (upper_shadow > 2 * body) &
        (lower_shadow < body) &
        is_bearish
    ).astype(int)

    # 4. ENGULFING (reversal)
    prev_body = body.shift(1)
    patterns['bullish_engulfing'] = (
        is_bullish &
        (df['close'] > df['open'].shift(1)) &
        (df['open'] < df['close'].shift(1)) &
        (body > prev_body)
    ).astype(int)

    patterns['bearish_engulfing'] = (
        is_bearish &
        (df['close'] < df['open'].shift(1)) &
        (df['open'] > df['close'].shift(1)) &
        (body > prev_body)
    ).astype(int)

    # 5. MARUBOZU (strong momentum)
    # No shadows, all body
    patterns['bullish_marubozu'] = (
        is_bullish &
        (upper_shadow < 0.01 * body) &
        (lower_shadow < 0.01 * body)
    ).astype(int)

    patterns['bearish_marubozu'] = (
        is_bearish &
        (upper_shadow < 0.01 * body) &
        (lower_shadow < 0.01 * body)
    ).astype(int)

    # 6. SPINNING TOP (indecision with shadows)
    patterns['spinning_top'] = (
        ((body / (total_range + 1e-10)) < 0.3) &
        (upper_shadow > body) &
        (lower_shadow > body)
    ).astype(int)

    return patterns


def detect_support_resistance_levels(df, window=20):
    """
    Détecte les niveaux de support et résistance via détection de pics.
    """
    features = {}

    # Find local peaks (resistance) and troughs (support)
    highs = df['high'].values
    lows = df['low'].values

    # Resistance levels (peaks in highs)
    resistance_indices, _ = find_peaks(highs, distance=window//2)

    # Support levels (peaks in inverted lows)
    support_indices, _ = find_peaks(-lows, distance=window//2)

    # Distance to nearest resistance
    features['dist_to_resistance'] = 0.0
    features['dist_to_support'] = 0.0

    for i in range(len(df)):
        current_price = df['close'].iloc[i]

        # Find nearest resistance above current price
        above_resistances = [highs[idx] for idx in resistance_indices if idx < i and highs[idx] > current_price]
        if above_resistances:
            features['dist_to_resistance'] = min(above_resistances) - current_price

        # Find nearest support below current price
        below_supports = [lows[idx] for idx in support_indices if idx < i and lows[idx] < current_price]
        if below_supports:
            features['dist_to_support'] = current_price - max(below_supports)

    # Normalize by price
    df['dist_to_resistance_pct'] = features['dist_to_resistance'] / (df['close'] + 1e-10)
    df['dist_to_support_pct'] = features['dist_to_support'] / (df['close'] + 1e-10)

    # Count nearby levels (clustering)
    df['resistance_cluster'] = 0
    df['support_cluster'] = 0

    for i in range(len(df)):
        current_price = df['close'].iloc[i]
        tolerance = 0.02  # 2% tolerance

        # Count resistances within 2%
        nearby_resistances = [idx for idx in resistance_indices
                            if idx < i and abs(highs[idx] - current_price) / current_price < tolerance]
        df.loc[df.index[i], 'resistance_cluster'] = len(nearby_resistances)

        # Count supports within 2%
        nearby_supports = [idx for idx in support_indices
                          if idx < i and abs(lows[idx] - current_price) / current_price < tolerance]
        df.loc[df.index[i], 'support_cluster'] = len(nearby_supports)

    return df


def detect_volatility_regime(df, n_clusters=3):
    """
    Détecte le régime de volatilité via K-Means clustering.

    Regimes: LOW, MEDIUM, HIGH volatility
    """
    # Calculate rolling volatility
    returns = df['close'].pct_change()
    volatility = returns.rolling(14).std()

    # Prepare for clustering (remove NaN)
    vol_values = volatility.dropna().values.reshape(-1, 1)

    # K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(vol_values)

    # Map clusters to regimes (sorted by volatility)
    cluster_means = [vol_values[clusters == i].mean() for i in range(n_clusters)]
    sorted_clusters = np.argsort(cluster_means)

    regime_map = {sorted_clusters[0]: 0,  # LOW
                  sorted_clusters[1]: 1,  # MEDIUM
                  sorted_clusters[2]: 2}  # HIGH

    # Create regime column
    df['volatility_regime'] = np.nan
    df.loc[volatility.dropna().index, 'volatility_regime'] = [regime_map[c] for c in clusters]

    # Forward fill NaN
    df['volatility_regime'] = df['volatility_regime'].fillna(method='ffill').fillna(1)

    # One-hot encode
    df['vol_regime_low'] = (df['volatility_regime'] == 0).astype(int)
    df['vol_regime_medium'] = (df['volatility_regime'] == 1).astype(int)
    df['vol_regime_high'] = (df['volatility_regime'] == 2).astype(int)

    return df


def detect_volume_profile(df):
    """
    Analyse le profil de volume pour détecter les zones de forte activité.
    """
    # Volume relative to average
    df['volume_relative'] = df['volume'] / df['volume'].rolling(20).mean()

    # Volume spike detection
    df['volume_spike'] = (df['volume_relative'] > 2.0).astype(int)

    # Volume trend (increasing/decreasing)
    df['volume_trend_7d'] = df['volume'].rolling(7).mean() / df['volume'].rolling(14).mean()
    df['volume_trend_30d'] = df['volume'].rolling(30).mean() / df['volume'].rolling(60).mean()

    # Price-Volume relationship
    returns = df['close'].pct_change()
    df['price_volume_corr_7d'] = returns.rolling(7).corr(df['volume'])
    df['price_volume_corr_30d'] = returns.rolling(30).corr(df['volume'])

    # Volume concentration (Gini-like)
    # High concentration = few large volume bars
    df['volume_concentration'] = df['volume'].rolling(20).apply(
        lambda x: np.sum((x - x.mean())**2) / (len(x) * x.var() + 1e-10)
    )

    return df


def detect_momentum_shifts(df):
    """
    Détecte les changements de momentum (acceleration/deceleration).
    """
    returns = df['close'].pct_change()

    # Momentum acceleration
    momentum_7 = returns.rolling(7).mean()
    momentum_14 = returns.rolling(14).mean()

    df['momentum_acceleration'] = momentum_7 - momentum_14

    # Momentum shift detection (zero-crossing)
    df['momentum_shift_bullish'] = ((momentum_7 > 0) & (momentum_7.shift(1) <= 0)).astype(int)
    df['momentum_shift_bearish'] = ((momentum_7 < 0) & (momentum_7.shift(1) >= 0)).astype(int)

    # Strength of momentum change
    df['momentum_change_strength'] = abs(momentum_7 - momentum_7.shift(1))

    return df


def detect_trend_strength(df):
    """
    Mesure la force et la qualité du trend.
    """
    # ADX-like indicator (Average Directional Index)
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift(1))
    low_close = abs(df['low'] - df['close'].shift(1))

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    plus_dm = (df['high'] - df['high'].shift(1)).clip(lower=0)
    minus_dm = (df['low'].shift(1) - df['low']).clip(lower=0)

    # Smooth
    tr_smooth = true_range.rolling(14).sum()
    plus_dm_smooth = plus_dm.rolling(14).sum()
    minus_dm_smooth = minus_dm.rolling(14).sum()

    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth

    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    df['trend_strength_adx'] = dx.rolling(14).mean()

    # Trend consistency (how often price follows trend)
    sma_50 = df['close'].rolling(50).mean()
    df['trend_consistency'] = (
        ((df['close'] > sma_50) & (df['close'].shift(1) > sma_50.shift(1))).astype(int) +
        ((df['close'] < sma_50) & (df['close'].shift(1) < sma_50.shift(1))).astype(int)
    ).rolling(20).mean()

    return df


def detect_volatility_breakout(df):
    """
    Détecte les breakouts de volatilité (expansion after contraction).
    """
    # Bollinger Bands width
    sma_20 = df['close'].rolling(20).mean()
    std_20 = df['close'].rolling(20).std()
    bb_width = 2 * std_20 / sma_20

    # Bollinger squeeze (low volatility)
    df['bb_squeeze'] = (bb_width < bb_width.rolling(50).quantile(0.2)).astype(int)

    # Volatility expansion
    df['volatility_expansion'] = (bb_width > bb_width.shift(1) * 1.2).astype(int)

    # Breakout detection (price breaks out of bollinger bands)
    bb_upper = sma_20 + 2 * std_20
    bb_lower = sma_20 - 2 * std_20

    df['breakout_upper'] = (df['close'] > bb_upper).astype(int)
    df['breakout_lower'] = (df['close'] < bb_lower).astype(int)

    return df


def detect_market_structure(df):
    """
    Détecte la structure de marché (higher highs, lower lows, etc).
    """
    # Higher highs / Lower lows
    df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
    df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)

    # Consecutive higher highs/lows (strength)
    df['consecutive_higher_highs'] = df['higher_high'].rolling(5).sum()
    df['consecutive_lower_lows'] = df['lower_low'].rolling(5).sum()

    # Market structure score (-1 to +1)
    # +1 = strong uptrend structure, -1 = strong downtrend structure
    df['market_structure_score'] = (
        df['consecutive_higher_highs'] - df['consecutive_lower_lows']
    ) / 5.0

    # Pivot points detection
    df['pivot_high'] = (
        (df['high'] > df['high'].shift(1)) &
        (df['high'] > df['high'].shift(-1))
    ).astype(int)

    df['pivot_low'] = (
        (df['low'] < df['low'].shift(1)) &
        (df['low'] < df['low'].shift(-1))
    ).astype(int)

    return df


def create_advanced_nontechnical_features(df):
    """
    Crée toutes les features non-techniques avancées.
    """
    logger.info("Creating advanced non-technical features...")

    # 1. Candlestick patterns
    patterns = detect_candlestick_patterns(df)
    for name, values in patterns.items():
        df[f'pattern_{name}'] = values

    # 2. Support/Resistance
    df = detect_support_resistance_levels(df)

    # 3. Volatility regime
    df = detect_volatility_regime(df)

    # 4. Volume profile
    df = detect_volume_profile(df)

    # 5. Momentum shifts
    df = detect_momentum_shifts(df)

    # 6. Trend strength
    df = detect_trend_strength(df)

    # 7. Volatility breakout
    df = detect_volatility_breakout(df)

    # 8. Market structure
    df = detect_market_structure(df)

    # Count new features
    nontechnical_features = [c for c in df.columns if any(x in c for x in [
        'pattern_', 'dist_to_', 'resistance_', 'support_', 'vol_regime_',
        'volume_', 'momentum_', 'trend_', 'bb_', 'breakout_', 'market_structure',
        'pivot_', 'higher_', 'lower_', 'consecutive_'
    ])]

    logger.info(f"  ✓ Added {len(nontechnical_features)} advanced non-technical features")

    return df


if __name__ == '__main__':
    """Test advanced features"""
    import sys
    from pathlib import Path

    # Load sample data
    for data_dir in [Path('data'), Path('data_processed')]:
        file_path = data_dir / 'ETH_1d.csv'
        if file_path.exists():
            df = pd.read_csv(file_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            break

    logger.info(f"Loaded {len(df)} rows")

    # Create features
    df = create_advanced_nontechnical_features(df)

    # Show sample
    logger.info(f"\nSample features (last 5 rows):")
    feature_cols = [c for c in df.columns if any(x in c for x in [
        'pattern_', 'dist_to_', 'vol_regime_', 'volume_spike', 'momentum_shift'
    ])]
    print(df[feature_cols[:10]].tail())

    logger.info(f"\n✓ SUCCESS: Created {len(feature_cols)} advanced features")
