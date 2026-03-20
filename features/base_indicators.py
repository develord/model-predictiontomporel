"""
V10 Base Indicators - Multi-Timeframe
======================================
Port of V9 base indicators with multi-timeframe support

Calculates 30 base technical indicators per timeframe:
- RSI (14, 21, overbought, oversold)
- MACD (line, signal, histogram, crossover)
- Bollinger Bands (upper, middle, lower, width, percent)
- EMA (12, 26, 50, 200, cross_12_26)
- SMA (20, 50, 200)
- ATR (14, atr_pct)
- Stochastic (K, D, overbought, oversold)
- ADX (14)
- OBV
- CMF (20)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate RSI indicator

    Args:
        series: Price series (close prices)
        period: RSI period (default 14)

    Returns:
        Series of RSI values (0-100)
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """
    Calculate EMA (Exponential Moving Average)

    Args:
        series: Price series
        period: EMA period

    Returns:
        Series of EMA values
    """
    return series.ewm(span=period, adjust=False).mean()


def calculate_sma(series: pd.Series, period: int) -> pd.Series:
    """
    Calculate SMA (Simple Moving Average)

    Args:
        series: Price series
        period: SMA period

    Returns:
        Series of SMA values
    """
    return series.rolling(window=period).mean()


def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
    """
    Calculate MACD indicator

    Args:
        series: Price series
        fast: Fast EMA period (default 12)
        slow: Slow EMA period (default 26)
        signal: Signal line period (default 9)

    Returns:
        Dict with 'macd', 'signal', 'histogram' Series
    """
    ema_fast = calculate_ema(series, fast)
    ema_slow = calculate_ema(series, slow)

    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line

    return {
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram
    }


def calculate_bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> Dict[str, pd.Series]:
    """
    Calculate Bollinger Bands

    Args:
        series: Price series
        period: Period for SMA and std dev (default 20)
        std_dev: Number of standard deviations (default 2)

    Returns:
        Dict with 'upper', 'middle', 'lower' Series
    """
    middle = calculate_sma(series, period)
    std = series.rolling(window=period).std()

    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)

    return {
        'upper': upper,
        'middle': middle,
        'lower': lower
    }


def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
    """
    Calculate Stochastic Oscillator

    Args:
        high: High price series
        low: Low price series
        close: Close price series
        k_period: %K period (default 14)
        d_period: %D period (default 3)

    Returns:
        Dict with 'k', 'd' Series
    """
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()

    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d = k.rolling(window=d_period).mean()

    return {
        'k': k,
        'd': d
    }


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate ATR (Average True Range)

    Args:
        high: High price series
        low: Low price series
        close: Close price series
        period: ATR period (default 14)

    Returns:
        Series of ATR values
    """
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()

    return atr


def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Calculate OBV (On-Balance Volume)

    Args:
        close: Close price series
        volume: Volume series

    Returns:
        Series of OBV values
    """
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    return obv


def calculate_cmf(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 20) -> pd.Series:
    """
    Calculate CMF (Chaikin Money Flow)

    Args:
        high: High price series
        low: Low price series
        close: Close price series
        volume: Volume series
        period: CMF period (default 20)

    Returns:
        Series of CMF values
    """
    mfm = ((close - low) - (high - close)) / (high - low)
    mfm = mfm.fillna(0)  # Handle division by zero

    mfv = mfm * volume
    cmf = mfv.rolling(window=period).sum() / volume.rolling(window=period).sum()

    return cmf


def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate ADX (Average Directional Index)

    Args:
        high: High price series
        low: Low price series
        close: Close price series
        period: ADX period (default 14)

    Returns:
        Series of ADX values
    """
    # Calculate +DM and -DM
    high_diff = high.diff()
    low_diff = -low.diff()

    plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
    minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)

    # Calculate ATR
    atr = calculate_atr(high, low, close, period)

    # Calculate +DI and -DI
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

    # Calculate DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()

    return adx


def calculate_base_indicators(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Calculate all 30 base indicators for a given timeframe

    Args:
        df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume', 'timestamp']
        timeframe: Timeframe identifier ('4h', '1d', '1w')

    Returns:
        DataFrame with original columns + 30 indicator columns (prefixed with timeframe)

    Indicator columns added (30 total):
        - {tf}_rsi_14, {tf}_rsi_21, {tf}_rsi_overbought, {tf}_rsi_oversold (4)
        - {tf}_macd_line, {tf}_macd_signal, {tf}_macd_histogram, {tf}_macd_crossover (4)
        - {tf}_bb_upper, {tf}_bb_middle, {tf}_bb_lower, {tf}_bb_width, {tf}_bb_percent (5)
        - {tf}_ema_12, {tf}_ema_26, {tf}_ema_50, {tf}_ema_200, {tf}_ema_cross_12_26 (5)
        - {tf}_sma_20, {tf}_sma_50, {tf}_sma_200 (3)
        - {tf}_atr_14, {tf}_atr_pct (2)
        - {tf}_stoch_k, {tf}_stoch_d, {tf}_stoch_overbought, {tf}_stoch_oversold (4)
        - {tf}_adx_14 (1)
        - {tf}_obv (1)
        - {tf}_cmf_20 (1)
    """
    result = df.copy()
    tf = timeframe

    # 1. RSI (4 features)
    result[f'{tf}_rsi_14'] = calculate_rsi(df['close'], 14)
    result[f'{tf}_rsi_21'] = calculate_rsi(df['close'], 21)
    result[f'{tf}_rsi_overbought'] = (result[f'{tf}_rsi_14'] > 70).astype(int)
    result[f'{tf}_rsi_oversold'] = (result[f'{tf}_rsi_14'] < 30).astype(int)

    # 2. MACD (4 features)
    macd = calculate_macd(df['close'])
    result[f'{tf}_macd_line'] = macd['macd']
    result[f'{tf}_macd_signal'] = macd['signal']
    result[f'{tf}_macd_histogram'] = macd['histogram']
    result[f'{tf}_macd_crossover'] = (
        (macd['macd'] > macd['signal']) &
        (macd['macd'].shift(1) <= macd['signal'].shift(1))
    ).astype(int)

    # 3. Bollinger Bands (5 features)
    bb = calculate_bollinger_bands(df['close'])
    result[f'{tf}_bb_upper'] = bb['upper']
    result[f'{tf}_bb_middle'] = bb['middle']
    result[f'{tf}_bb_lower'] = bb['lower']
    result[f'{tf}_bb_width'] = (bb['upper'] - bb['lower']) / bb['middle']
    result[f'{tf}_bb_percent'] = (df['close'] - bb['lower']) / (bb['upper'] - bb['lower'])

    # 4. EMA (5 features)
    result[f'{tf}_ema_12'] = calculate_ema(df['close'], 12)
    result[f'{tf}_ema_26'] = calculate_ema(df['close'], 26)
    result[f'{tf}_ema_50'] = calculate_ema(df['close'], 50)
    result[f'{tf}_ema_200'] = calculate_ema(df['close'], 200)
    result[f'{tf}_ema_cross_12_26'] = (
        (result[f'{tf}_ema_12'] > result[f'{tf}_ema_26']) &
        (result[f'{tf}_ema_12'].shift(1) <= result[f'{tf}_ema_26'].shift(1))
    ).astype(int)

    # 5. SMA (3 features)
    result[f'{tf}_sma_20'] = calculate_sma(df['close'], 20)
    result[f'{tf}_sma_50'] = calculate_sma(df['close'], 50)
    result[f'{tf}_sma_200'] = calculate_sma(df['close'], 200)

    # 6. ATR (2 features)
    atr = calculate_atr(df['high'], df['low'], df['close'])
    result[f'{tf}_atr_14'] = atr
    result[f'{tf}_atr_pct'] = (atr / df['close']) * 100

    # 7. Stochastic (4 features)
    stoch = calculate_stochastic(df['high'], df['low'], df['close'])
    result[f'{tf}_stoch_k'] = stoch['k']
    result[f'{tf}_stoch_d'] = stoch['d']
    result[f'{tf}_stoch_overbought'] = (stoch['k'] > 80).astype(int)
    result[f'{tf}_stoch_oversold'] = (stoch['k'] < 20).astype(int)

    # 8. ADX (1 feature)
    result[f'{tf}_adx_14'] = calculate_adx(df['high'], df['low'], df['close'])

    # 9. OBV (1 feature)
    result[f'{tf}_obv'] = calculate_obv(df['close'], df['volume'])

    # 10. CMF (1 feature)
    result[f'{tf}_cmf_20'] = calculate_cmf(df['high'], df['low'], df['close'], df['volume'])

    return result


def calculate_multi_tf_base_indicators(
    df_4h: pd.DataFrame,
    df_1d: pd.DataFrame,
    df_1w: pd.DataFrame
) -> Dict[str, pd.DataFrame]:
    """
    Calculate base indicators for all 3 timeframes

    Args:
        df_4h: 4-hour dataframe
        df_1d: 1-day dataframe
        df_1w: 1-week dataframe

    Returns:
        Dict with keys '4h', '1d', '1w' containing DataFrames with indicators

    Feature count:
        - Per timeframe: 30 indicators
        - Total: 90 base indicators (30 × 3 timeframes)
    """
    return {
        '4h': calculate_base_indicators(df_4h, '4h'),
        '1d': calculate_base_indicators(df_1d, '1d'),
        '1w': calculate_base_indicators(df_1w, '1w')
    }
