"""
ENHANCED FEATURES ENRICHED - COMPREHENSIVE NON-TECHNICAL FEATURE SYSTEM
========================================================================

Combines ALL beneficial features for multi-timeframe prediction:
1. Base technical features (~90)
2. BTC correlation features (~30 for non-BTC cryptos)
3. Regime/market state features (~10)
4. Time-based features (~10)
5. Volume analysis features (~10)
6. Market structure features (~5)
7. Advanced non-technical features (~38):
   - Candlestick patterns (10)
   - Support/Resistance levels (4)
   - Volatility regime detection (3)
   - Volume profile analysis (7)
   - Momentum shifts (4)
   - Trend strength indicators (2)
   - Volatility breakout detection (4)
   - Market structure analysis (8)

Total: ~134 features per timeframe

Author: Advanced Trading System
Date: 2026-03-29
Version: 3.0 - Ultimate ETH XGBoost with Advanced Non-Technical Features
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List
import logging
from advanced_features_nontechnical import create_advanced_nontechnical_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_btc_data(timeframe: str = '1d') -> pd.DataFrame:
    """
    Charge données BTC pour le timeframe donné.

    Args:
        timeframe: '1h', '4h', ou '1d'

    Returns:
        DataFrame BTC avec colonnes OHLCV + timestamp
    """
    # Try both data/ and data_processed/
    for data_dir in [Path('data_processed'), Path('data'), Path('data/cache')]:
        btc_file = data_dir / f'BTC_{timeframe}.csv'
        if btc_file.exists():
            df = pd.read_csv(btc_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            return df

    raise FileNotFoundError(f"BTC data not found for {timeframe} in data/ or data_processed/")


def create_base_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create base technical indicators (existing features from enhanced_features_fixed.py).
    This is a placeholder - in real implementation, would import from enhanced_features_fixed.

    Returns ~90 technical features.
    """
    df = df.copy()

    # Returns
    df['returns_1'] = df['close'].pct_change(1)
    df['returns_7'] = df['close'].pct_change(7)
    df['returns_14'] = df['close'].pct_change(14)
    df['returns_30'] = df['close'].pct_change(30)

    # Moving Averages
    for period in [7, 14, 30, 50]:
        df[f'sma_{period}'] = df['close'].rolling(period).mean()
        df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()

    # Price relative to MAs
    for period in [7, 14, 30]:
        df[f'price_to_sma_{period}'] = (df['close'] / df[f'sma_{period}']) - 1

    # RSI
    for period in [14, 30]:
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-10)
        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    for period in [20, 50]:
        sma = df['close'].rolling(period).mean()
        std = df['close'].rolling(period).std()
        df[f'bb_upper_{period}'] = sma + (2 * std)
        df[f'bb_lower_{period}'] = sma - (2 * std)
        df[f'bb_width_{period}'] = (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / sma
        df[f'bb_position_{period}'] = (df['close'] - df[f'bb_lower_{period}']) / (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'] + 1e-10)

    # MACD
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # ATR (volatility)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    for period in [14, 30]:
        df[f'atr_{period}'] = true_range.rolling(period).mean()
        df[f'atr_{period}_pct'] = df[f'atr_{period}'] / df['close']

    # Volume features
    df['volume_sma_20'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_sma_20'] + 1e-10)
    df['volume_ema_10'] = df['volume'].ewm(span=10, adjust=False).mean()

    # Momentum
    for period in [7, 14, 30]:
        df[f'momentum_{period}'] = (df['close'] / df['close'].shift(period)) - 1

    # Volatility (rolling std of returns)
    for period in [7, 14, 30]:
        df[f'volatility_{period}'] = df['returns_1'].rolling(period).std()

    # High-Low range
    df['high_low_range'] = (df['high'] - df['low']) / df['close']

    return df


def create_btc_correlation_features(df_crypto: pd.DataFrame, df_btc: pd.DataFrame, crypto_symbol: str = 'ETH') -> pd.DataFrame:
    """
    Crée features de correlation BTC (~30 features).

    Features:
    1. Price Ratio (ETH/BTC ratio + SMAs + trend)
    2. Returns Correlation (rolling 7d, 14d, 30d)
    3. Relative Strength (performance difference vs BTC)
    4. Volume Ratio (ETH volume / BTC volume)
    5. Volatility Ratio (ETH vol / BTC vol)
    6. BTC Regime Features (trend, momentum, bull/bear/sideways)
    7. Beta (regression-like ETH vs BTC)
    """
    logger.info(f"Creating BTC correlation features for {crypto_symbol}...")

    df = df_crypto.copy()
    df_btc = df_btc.copy()

    # Ensure timestamp is datetime for both dataframes
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df_btc['timestamp'] = pd.to_datetime(df_btc['timestamp'])

    # Renommer colonnes BTC
    btc_cols = {
        'close': 'btc_close',
        'open': 'btc_open',
        'high': 'btc_high',
        'low': 'btc_low',
        'volume': 'btc_volume'
    }
    df_btc = df_btc.rename(columns=btc_cols)
    df_btc = df_btc[['timestamp', 'btc_close', 'btc_open', 'btc_high', 'btc_low', 'btc_volume']]

    # Merge
    df = pd.merge(df, df_btc, on='timestamp', how='left')

    # Fill missing BTC data (forward fill)
    for col in ['btc_close', 'btc_volume', 'btc_high', 'btc_low']:
        df[col] = df[col].fillna(method='ffill')

    # =====================
    # 1. PRICE RATIO
    # =====================
    df['crypto_btc_ratio'] = df['close'] / df['btc_close']
    df['crypto_btc_ratio_sma_7'] = df['crypto_btc_ratio'].rolling(7).mean()
    df['crypto_btc_ratio_sma_30'] = df['crypto_btc_ratio'].rolling(30).mean()
    df['crypto_btc_ratio_trend'] = (df['crypto_btc_ratio_sma_7'] / df['crypto_btc_ratio_sma_30']) - 1

    # =====================
    # 2. RETURNS CORRELATION
    # =====================
    df['crypto_return'] = df['close'].pct_change()
    df['btc_return'] = df['btc_close'].pct_change()

    # Rolling correlation
    df['corr_btc_7d'] = df['crypto_return'].rolling(7).corr(df['btc_return'])
    df['corr_btc_14d'] = df['crypto_return'].rolling(14).corr(df['btc_return'])
    df['corr_btc_30d'] = df['crypto_return'].rolling(30).corr(df['btc_return'])

    # =====================
    # 3. RELATIVE STRENGTH
    # =====================
    for window in [7, 14, 30]:
        crypto_perf = (df['close'] / df['close'].shift(window)) - 1
        btc_perf = (df['btc_close'] / df['btc_close'].shift(window)) - 1
        df[f'rel_strength_{window}d'] = crypto_perf - btc_perf

    # =====================
    # 4. VOLUME RELATIF
    # =====================
    df['volume_ratio_btc'] = df['volume'] / (df['btc_volume'] + 1e-10)
    df['volume_ratio_btc_sma_7'] = df['volume_ratio_btc'].rolling(7).mean()

    # =====================
    # 5. VOLATILITY RATIO
    # =====================
    df['crypto_volatility_7d'] = df['crypto_return'].rolling(7).std()
    df['btc_volatility_7d'] = df['btc_return'].rolling(7).std()
    df['volatility_ratio_btc'] = df['crypto_volatility_7d'] / (df['btc_volatility_7d'] + 1e-10)

    df['crypto_volatility_30d'] = df['crypto_return'].rolling(30).std()
    df['btc_volatility_30d'] = df['btc_return'].rolling(30).std()
    df['volatility_ratio_btc_30d'] = df['crypto_volatility_30d'] / (df['btc_volatility_30d'] + 1e-10)

    # =====================
    # 6. BTC REGIME FEATURES
    # =====================
    df['btc_sma_7'] = df['btc_close'].rolling(7).mean()
    df['btc_sma_30'] = df['btc_close'].rolling(30).mean()
    df['btc_trend_short'] = (df['btc_close'] / df['btc_sma_7']) - 1
    df['btc_trend_long'] = (df['btc_sma_7'] / df['btc_sma_30']) - 1

    # BTC Momentum
    df['btc_momentum_7d'] = (df['btc_close'] / df['btc_close'].shift(7)) - 1
    df['btc_momentum_14d'] = (df['btc_close'] / df['btc_close'].shift(14)) - 1
    df['btc_momentum_30d'] = (df['btc_close'] / df['btc_close'].shift(30)) - 1

    # BTC Regime classification
    vol_median = df['btc_volatility_30d'].median()
    df['btc_regime_bull'] = ((df['btc_momentum_30d'] > 0) & (df['btc_volatility_30d'] < vol_median)).astype(int)
    df['btc_regime_bear'] = ((df['btc_momentum_30d'] < 0) & (df['btc_volatility_30d'] > vol_median)).astype(int)
    df['btc_regime_sideways'] = ((df['btc_regime_bull'] == 0) & (df['btc_regime_bear'] == 0)).astype(int)

    # =====================
    # 7. BETA
    # =====================
    for window in [7, 30]:
        cov = df['crypto_return'].rolling(window).cov(df['btc_return'])
        var = df['btc_return'].rolling(window).var()
        df[f'beta_btc_{window}d'] = cov / (var + 1e-10)

    # Cleanup - drop intermediate columns
    cols_to_drop = ['btc_close', 'btc_open', 'btc_high', 'btc_low', 'btc_volume',
                    'crypto_return', 'btc_return',
                    'crypto_volatility_7d', 'btc_volatility_7d',
                    'crypto_volatility_30d', 'btc_volatility_30d',
                    'btc_sma_7', 'btc_sma_30']

    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Count features added
    btc_features = [c for c in df.columns if any(x in c.lower() for x in
                    ['btc', 'corr', 'ratio', 'rel_strength', 'volume_ratio', 'volatility_ratio', 'beta', 'regime'])]

    logger.info(f"  Added {len(btc_features)} BTC correlation features")

    return df


def create_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée features de régime de marché (~10 features).

    Features:
    1. Market regime (bull/bear/sideways based on returns + volatility)
    2. Trend strength
    3. Volatility regime (low/medium/high)
    """
    df = df.copy()

    # Market returns for regime detection
    df['return_30d_regime'] = df['close'].pct_change(30)
    df['volatility_30d_regime'] = df['returns_1'].rolling(30).std()

    # Regime classification (similar to REGIME-BASED approach)
    vol_median = df['volatility_30d_regime'].median()

    df['regime_bull'] = ((df['return_30d_regime'] > 0) & (df['volatility_30d_regime'] < vol_median)).astype(int)
    df['regime_bear'] = ((df['return_30d_regime'] < 0) & (df['volatility_30d_regime'] > vol_median)).astype(int)
    df['regime_sideways'] = ((df['regime_bull'] == 0) & (df['regime_bear'] == 0)).astype(int)

    # Trend strength (how strong is the trend?)
    df['trend_strength_7d'] = abs(df['returns_7'])
    df['trend_strength_30d'] = abs(df['return_30d_regime'])

    # Volatility regime (low/medium/high)
    vol_33 = df['volatility_30d_regime'].quantile(0.33)
    vol_67 = df['volatility_30d_regime'].quantile(0.67)

    df['vol_regime_low'] = (df['volatility_30d_regime'] < vol_33).astype(int)
    df['vol_regime_medium'] = ((df['volatility_30d_regime'] >= vol_33) & (df['volatility_30d_regime'] < vol_67)).astype(int)
    df['vol_regime_high'] = (df['volatility_30d_regime'] >= vol_67).astype(int)

    # Cleanup
    df = df.drop(columns=['return_30d_regime', 'volatility_30d_regime'])

    return df


def create_time_based_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée features temporelles (~10 features).

    Features:
    1. Day of week (cyclical encoding)
    2. Hour of day (if applicable for 1h/4h)
    3. Day of month
    4. Week of year
    5. Month seasonality
    """
    df = df.copy()

    # Ensure timestamp is datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Day of week (0=Monday, 6=Sunday) - cyclical encoding
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    # Hour of day (if applicable) - cyclical encoding
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['hour_of_day_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
    df['hour_of_day_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)

    # Day of month - cyclical encoding
    df['day_of_month'] = df['timestamp'].dt.day
    df['day_of_month_sin'] = np.sin(2 * np.pi * df['day_of_month'] / 31)
    df['day_of_month_cos'] = np.cos(2 * np.pi * df['day_of_month'] / 31)

    # Month - cyclical encoding (seasonality)
    df['month'] = df['timestamp'].dt.month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Weekend indicator
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    # Cleanup intermediate columns
    df = df.drop(columns=['day_of_week', 'hour_of_day', 'day_of_month', 'month'])

    return df


def create_volume_analysis_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée features d'analyse de volume (~10 features).

    Features:
    1. Volume trends (increasing/decreasing)
    2. Volume breakouts
    3. Volume divergence from price
    4. On-Balance Volume (OBV)
    """
    df = df.copy()

    # Volume trend
    df['volume_trend_7d'] = df['volume'].rolling(7).mean() / (df['volume'].rolling(30).mean() + 1e-10)
    df['volume_trend_14d'] = df['volume'].rolling(14).mean() / (df['volume'].rolling(30).mean() + 1e-10)

    # Volume breakout (volume significantly above average)
    vol_std = df['volume'].rolling(20).std()
    vol_mean = df['volume'].rolling(20).mean()
    df['volume_breakout'] = ((df['volume'] - vol_mean) / (vol_std + 1e-10)).clip(-3, 3)

    # Volume divergence (price up but volume down, or vice versa)
    price_change_7d = df['close'].pct_change(7)
    volume_change_7d = df['volume'].pct_change(7)
    df['volume_price_divergence_7d'] = price_change_7d * volume_change_7d  # Negative = divergence

    # On-Balance Volume (OBV)
    obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['obv_7d_slope'] = obv.rolling(7).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 7 else 0, raw=True)
    df['obv_14d_slope'] = obv.rolling(14).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 14 else 0, raw=True)

    # Volume concentration (how much volume in recent bars vs historical)
    df['volume_concentration'] = df['volume'].rolling(7).sum() / (df['volume'].rolling(30).sum() + 1e-10)

    return df


def create_market_structure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée features de structure de marché (~5 features).

    Features:
    1. Support/resistance levels (price near support/resistance)
    2. Range vs trending market
    3. Higher highs / Lower lows patterns
    """
    df = df.copy()

    # Range vs trending
    # Range: price oscillates within bounds
    # Trending: price consistently moving in one direction
    high_30 = df['high'].rolling(30).max()
    low_30 = df['low'].rolling(30).min()
    range_size = (high_30 - low_30) / df['close']

    df['market_range_30d'] = range_size
    df['price_position_in_range'] = (df['close'] - low_30) / (high_30 - low_30 + 1e-10)

    # Distance from 30d high/low (potential support/resistance)
    df['distance_from_high_30d'] = (df['close'] - high_30) / high_30
    df['distance_from_low_30d'] = (df['close'] - low_30) / low_30

    # Higher highs / Lower lows pattern
    df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
    df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)

    # Consecutive higher highs / lower lows (trend strength)
    df['consecutive_higher_highs'] = df['higher_high'].rolling(7).sum()
    df['consecutive_lower_lows'] = df['lower_low'].rolling(7).sum()

    # Cleanup
    df = df.drop(columns=['higher_high', 'lower_low'])

    return df


def create_enriched_features(df: pd.DataFrame, crypto_symbol: str = 'ETH', timeframe: str = '1d') -> pd.DataFrame:
    """
    Create comprehensive enriched features combining ALL beneficial feature types.

    Args:
        df: DataFrame with OHLCV data
        crypto_symbol: 'ETH', 'SOL', etc.
        timeframe: '1h', '4h', '1d'

    Returns:
        DataFrame with ~134 features (96 base + 38 advanced non-technical)
    """
    logger.info(f"Creating enriched features for {crypto_symbol} {timeframe}...")

    # 1. Base technical features (~90)
    df = create_base_technical_features(df)
    logger.info(f"  Base technical features created")

    # 2. BTC correlation features (~30) - only for non-BTC cryptos
    if crypto_symbol.upper() != 'BTC':
        try:
            df_btc = load_btc_data(timeframe=timeframe)
            df = create_btc_correlation_features(df, df_btc, crypto_symbol)
        except FileNotFoundError as e:
            logger.warning(f"  Could not load BTC data: {e}")
            logger.warning(f"  Skipping BTC correlation features")

    # 3. Regime features (~10)
    df = create_regime_features(df)
    logger.info(f"  Regime features created")

    # 4. Time-based features (~10)
    df = create_time_based_features(df)
    logger.info(f"  Time-based features created")

    # 5. Volume analysis features (~10)
    df = create_volume_analysis_features(df)
    logger.info(f"  Volume analysis features created")

    # 6. Market structure features (~5)
    df = create_market_structure_features(df)
    logger.info(f"  Market structure features created")

    # 7. Advanced non-technical features (~38)
    df = create_advanced_nontechnical_features(df)
    logger.info(f"  Advanced non-technical features created")

    # Total features
    feature_cols = [c for c in df.columns if c not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    logger.info(f"  Total features created: {len(feature_cols)}")

    return df


def create_enriched_features_for_training(
    crypto_symbol: str,
    timeframes: List[str] = ['1h', '4h', '1d']
) -> dict:
    """
    Crée features enrichies pour tous les timeframes d'une crypto.

    Args:
        crypto_symbol: 'ETH', 'SOL', etc.
        timeframes: Liste des timeframes à traiter

    Returns:
        dict: {timeframe: df_with_enriched_features}
    """
    result = {}

    for tf in timeframes:
        logger.info(f"\nProcessing {crypto_symbol} {tf}...")

        # Load crypto data
        crypto_file = None
        for data_dir in [Path('data_processed'), Path('data'), Path('data/cache')]:
            test_file = data_dir / f'{crypto_symbol}_{tf}.csv'
            if test_file.exists():
                crypto_file = test_file
                break

        if crypto_file is None:
            logger.warning(f"  File not found: {crypto_symbol}_{tf}.csv")
            continue

        df_crypto = pd.read_csv(crypto_file)
        df_crypto['timestamp'] = pd.to_datetime(df_crypto['timestamp'])

        # Create enriched features
        df_enriched = create_enriched_features(df_crypto, crypto_symbol, timeframe=tf)

        result[tf] = df_enriched
        logger.info(f"  {crypto_symbol} {tf}: {len(df_enriched)} rows, {len(df_enriched.columns)} features")

    return result


if __name__ == '__main__':
    """Test enriched features creation"""
    import sys

    logger.info("=" * 80)
    logger.info("TESTING ENRICHED FEATURES SYSTEM")
    logger.info("=" * 80)

    try:
        # Test sur ETH 1d
        result = create_enriched_features_for_training('ETH', timeframes=['1d'])

        if '1d' in result:
            df = result['1d']
            logger.info(f"\nSample features (last 5 rows):")
            feature_cols = [c for c in df.columns if c not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            logger.info(f"Total feature columns: {len(feature_cols)}")

            # Show feature categories
            btc_features = [c for c in feature_cols if 'btc' in c.lower() or 'corr' in c.lower() or 'ratio' in c.lower() or 'rel_strength' in c.lower()]
            regime_features = [c for c in feature_cols if 'regime' in c.lower()]
            time_features = [c for c in feature_cols if any(x in c.lower() for x in ['sin', 'cos', 'weekend', 'day', 'hour', 'month'])]
            volume_features = [c for c in feature_cols if 'volume' in c.lower() or 'obv' in c.lower()]
            structure_features = [c for c in feature_cols if any(x in c.lower() for x in ['range', 'distance', 'higher', 'lower', 'position'])]

            logger.info(f"\nFeature breakdown:")
            logger.info(f"  BTC correlation: {len(btc_features)}")
            logger.info(f"  Regime: {len(regime_features)}")
            logger.info(f"  Time-based: {len(time_features)}")
            logger.info(f"  Volume analysis: {len(volume_features)}")
            logger.info(f"  Market structure: {len(structure_features)}")
            logger.info(f"  Other technical: {len(feature_cols) - len(btc_features) - len(regime_features) - len(time_features) - len(volume_features) - len(structure_features)}")

            logger.info(f"\nSUCCESS: Created {len(feature_cols)} enriched features for ETH")
        else:
            logger.error("Failed to create enriched features")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
