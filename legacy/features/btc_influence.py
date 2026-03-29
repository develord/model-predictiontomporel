"""
V10 BTC Influence Features - Multi-Timeframe
=============================================
Port of V9 BTC influence features with multi-timeframe support

Calculates 37 BTC influence features per timeframe (ETH/SOL only):
- BTC price correlation (4 windows): 4 features
- BTC momentum influence (3 periods): 6 features
- Relative strength vs BTC (3 periods): 3 features
- BTC volume influence (3 periods): 6 features
- BTC trend following: 4 features
- BTC dominance pattern (2 periods): 4 features
- BTC volatility influence (3 windows): 6 features
- BTC divergence signals: 4 features

V9 Impact: ETH 0% → +70.51% ROI! (CRITICAL feature set)
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional


def calculate_btc_price_correlation(
    df_asset: pd.DataFrame,
    df_btc: pd.DataFrame,
    timeframe: str,
    windows: list = [7, 14, 30, 60]
) -> pd.DataFrame:
    """
    BTC price correlation over multiple windows

    Args:
        df_asset: Asset (ETH/SOL) dataframe
        df_btc: BTC dataframe (same timeframe)
        timeframe: Timeframe identifier
        windows: Correlation windows

    Returns:
        DataFrame with 4 correlation features
    """
    result = df_asset.copy()
    tf = timeframe

    for window in windows:
        result[f'{tf}_btc_corr_{window}'] = df_asset['close'].rolling(window=window).corr(
            df_btc['close']
        ).fillna(0)

    return result


def calculate_btc_momentum_influence(
    df_asset: pd.DataFrame,
    df_btc: pd.DataFrame,
    timeframe: str,
    periods: list = [3, 7, 14]
) -> pd.DataFrame:
    """
    BTC momentum influence on asset

    Args:
        df_asset: Asset dataframe
        df_btc: BTC dataframe
        timeframe: Timeframe identifier
        periods: Momentum periods

    Returns:
        DataFrame with 6 features (BTC momentum + ratio per period)
    """
    result = df_asset.copy()
    tf = timeframe

    for period in periods:
        # BTC momentum
        btc_momentum = df_btc['close'].pct_change(period) * 100
        result[f'{tf}_btc_momentum_{period}'] = btc_momentum.fillna(0)

        # Asset momentum
        asset_momentum = df_asset['close'].pct_change(period) * 100

        # Momentum ratio (how much asset moves relative to BTC)
        result[f'{tf}_btc_momentum_ratio_{period}'] = (
            asset_momentum / btc_momentum.replace(0, np.nan)
        ).fillna(0).replace([np.inf, -np.inf], 0)

    return result


def calculate_relative_strength_vs_btc(
    df_asset: pd.DataFrame,
    df_btc: pd.DataFrame,
    timeframe: str,
    periods: list = [7, 14, 30]
) -> pd.DataFrame:
    """
    Relative strength of asset vs BTC (outperformance/underperformance)

    Args:
        df_asset: Asset dataframe
        df_btc: BTC dataframe
        timeframe: Timeframe identifier
        periods: Performance periods

    Returns:
        DataFrame with 3 relative strength features
    """
    result = df_asset.copy()
    tf = timeframe

    for period in periods:
        # Asset change
        asset_change = df_asset['close'].pct_change(period)

        # BTC change
        btc_change = df_btc['close'].pct_change(period)

        # Relative strength (positive = outperforming BTC)
        result[f'{tf}_btc_relative_strength_{period}'] = (asset_change - btc_change).fillna(0)

    return result


def calculate_btc_volume_influence(
    df_asset: pd.DataFrame,
    df_btc: pd.DataFrame,
    timeframe: str,
    periods: list = [3, 7, 14]
) -> pd.DataFrame:
    """
    BTC volume influence on asset

    Args:
        df_asset: Asset dataframe
        df_btc: BTC dataframe
        timeframe: Timeframe identifier
        periods: Volume periods

    Returns:
        DataFrame with 6 features (volume correlation + ratio per period)
    """
    result = df_asset.copy()
    tf = timeframe

    for period in periods:
        # Volume correlation
        result[f'{tf}_btc_volume_corr_{period}'] = df_asset['volume'].rolling(window=period).corr(
            df_btc['volume']
        ).fillna(0)

        # Volume ratio (asset volume relative to BTC)
        avg_asset_vol = df_asset['volume'].rolling(window=period).mean()
        avg_btc_vol = df_btc['volume'].rolling(window=period).mean()

        result[f'{tf}_btc_volume_ratio_{period}'] = (
            avg_asset_vol / avg_btc_vol.replace(0, np.nan)
        ).fillna(0).replace([np.inf, -np.inf], 0)

    return result


def calculate_btc_trend_following(
    df_asset: pd.DataFrame,
    df_btc: pd.DataFrame,
    timeframe: str,
    window: int = 14
) -> pd.DataFrame:
    """
    BTC trend following patterns

    Args:
        df_asset: Asset dataframe
        df_btc: BTC dataframe
        timeframe: Timeframe identifier
        window: Trend window

    Returns:
        DataFrame with 4 trend features
    """
    result = df_asset.copy()
    tf = timeframe

    # BTC trend (linear regression slope)
    def calc_trend(x):
        if len(x) < 2:
            return 0
        y = np.array(x)
        slope = np.polyfit(np.arange(len(y)), y, 1)[0]
        return (slope / np.mean(y) * 100) if np.mean(y) > 0 else 0

    btc_trend = df_btc['close'].rolling(window=window).apply(calc_trend, raw=True)
    asset_trend = df_asset['close'].rolling(window=window).apply(calc_trend, raw=True)

    result[f'{tf}_btc_trend'] = btc_trend.fillna(0)
    result[f'{tf}_asset_trend'] = asset_trend.fillna(0)

    # Trend alignment (both up, both down, or diverging)
    result[f'{tf}_btc_trend_alignment'] = (
        (btc_trend * asset_trend > 0).astype(int)
    )

    # Trend strength difference
    result[f'{tf}_btc_trend_diff'] = (asset_trend - btc_trend).fillna(0)

    return result


def calculate_btc_dominance_pattern(
    df_asset: pd.DataFrame,
    df_btc: pd.DataFrame,
    timeframe: str,
    periods: list = [7, 14]
) -> pd.DataFrame:
    """
    BTC dominance patterns (altcoin season detection)

    Args:
        df_asset: Asset dataframe
        df_btc: BTC dataframe
        timeframe: Timeframe identifier
        periods: Dominance periods

    Returns:
        DataFrame with 4 features (dominance + altcoin season per period)
    """
    result = df_asset.copy()
    tf = timeframe

    for period in periods:
        # Returns
        btc_return = df_btc['close'].pct_change(period) * 100
        asset_return = df_asset['close'].pct_change(period) * 100

        # BTC dominance (BTC outperforming = altcoin season ending)
        result[f'{tf}_btc_dominance_{period}'] = (btc_return - asset_return).fillna(0)

        # Altcoin season indicator (asset outperforming = 1)
        result[f'{tf}_altcoin_season_{period}'] = (asset_return > btc_return).astype(int)

    return result


def calculate_btc_volatility_influence(
    df_asset: pd.DataFrame,
    df_btc: pd.DataFrame,
    timeframe: str,
    windows: list = [7, 14, 30]
) -> pd.DataFrame:
    """
    BTC volatility influence on asset

    Args:
        df_asset: Asset dataframe
        df_btc: BTC dataframe
        timeframe: Timeframe identifier
        windows: Volatility windows

    Returns:
        DataFrame with 6 features (BTC volatility + ratio per window)
    """
    result = df_asset.copy()
    tf = timeframe

    for window in windows:
        # BTC volatility (coefficient of variation)
        btc_mean = df_btc['close'].rolling(window=window).mean()
        btc_std = df_btc['close'].rolling(window=window).std()
        btc_vol = (btc_std / btc_mean * 100).fillna(0)

        result[f'{tf}_btc_volatility_{window}'] = btc_vol

        # Asset volatility
        asset_mean = df_asset['close'].rolling(window=window).mean()
        asset_std = df_asset['close'].rolling(window=window).std()
        asset_vol = (asset_std / asset_mean * 100).fillna(0)

        # Volatility ratio (asset vol / BTC vol)
        result[f'{tf}_btc_volatility_ratio_{window}'] = (
            asset_vol / btc_vol.replace(0, np.nan)
        ).fillna(1.0).replace([np.inf, -np.inf], 1.0)

    return result


def calculate_btc_divergence_signals(
    df_asset: pd.DataFrame,
    df_btc: pd.DataFrame,
    timeframe: str,
    window: int = 14
) -> pd.DataFrame:
    """
    BTC divergence signals (price-volume divergences)

    Args:
        df_asset: Asset dataframe
        df_btc: BTC dataframe
        timeframe: Timeframe identifier
        window: Divergence window

    Returns:
        DataFrame with 4 divergence features
    """
    result = df_asset.copy()
    tf = timeframe

    # Price trends
    btc_price_trend = df_btc['close'].pct_change(window)
    asset_price_trend = df_asset['close'].pct_change(window)

    # Volume trends (compare early vs late half of window)
    def calc_volume_trend(x):
        if len(x) < 2:
            return 0
        mid = len(x) // 2
        early = x[:mid].mean()
        late = x[mid:].mean()
        if early == 0:
            return 0
        return (late - early) / early

    btc_vol_trend = df_btc['volume'].rolling(window=window).apply(calc_volume_trend, raw=True)
    asset_vol_trend = df_asset['volume'].rolling(window=window).apply(calc_volume_trend, raw=True)

    # BTC price-volume divergence
    result[f'{tf}_btc_price_vol_divergence'] = (
        (np.sign(btc_price_trend) != np.sign(btc_vol_trend)).astype(int)
    )

    # Asset price-volume divergence
    result[f'{tf}_asset_price_vol_divergence'] = (
        (np.sign(asset_price_trend) != np.sign(asset_vol_trend)).astype(int)
    )

    # Cross-asset divergence (BTC up, asset down or vice versa)
    result[f'{tf}_btc_asset_divergence'] = (
        (np.sign(btc_price_trend) != np.sign(asset_price_trend)).astype(int)
    )

    # Synchronized movement (both moving same direction)
    result[f'{tf}_btc_asset_synchronized'] = (
        (btc_price_trend * asset_price_trend > 0).astype(int)
    )

    return result


def calculate_btc_influence_features(
    df_asset: pd.DataFrame,
    df_btc: pd.DataFrame,
    timeframe: str,
    crypto_symbol: str
) -> pd.DataFrame:
    """
    Calculate all 37 BTC influence features for a given timeframe

    Args:
        df_asset: Asset (ETH/SOL) dataframe
        df_btc: BTC dataframe (same timeframe, same length)
        timeframe: Timeframe identifier ('4h', '1d', '1w')
        crypto_symbol: 'ETHUSDT' or 'SOLUSDT'

    Returns:
        DataFrame with original columns + 37 BTC influence features
        Returns df_asset unchanged if crypto_symbol is BTC

    Feature breakdown:
        - BTC price correlation: 4
        - BTC momentum influence: 6
        - Relative strength vs BTC: 3
        - BTC volume influence: 6
        - BTC trend following: 4
        - BTC dominance pattern: 4
        - BTC volatility influence: 6
        - BTC divergence signals: 4
        TOTAL: 37 features
    """
    # If this IS BTC, return unchanged (no self-influence)
    if crypto_symbol == 'BTCUSDT':
        return df_asset

    # Ensure dataframes are aligned
    if len(df_asset) != len(df_btc):
        min_len = min(len(df_asset), len(df_btc))
        df_asset = df_asset.iloc[-min_len:].copy()
        df_btc = df_btc.iloc[-min_len:].copy()

    result = df_asset.copy()

    # 1. BTC price correlation (4)
    result = calculate_btc_price_correlation(result, df_btc, timeframe)

    # 2. BTC momentum influence (6)
    result = calculate_btc_momentum_influence(result, df_btc, timeframe)

    # 3. Relative strength vs BTC (3)
    result = calculate_relative_strength_vs_btc(result, df_btc, timeframe)

    # 4. BTC volume influence (6)
    result = calculate_btc_volume_influence(result, df_btc, timeframe)

    # 5. BTC trend following (4)
    result = calculate_btc_trend_following(result, df_btc, timeframe)

    # 6. BTC dominance pattern (4)
    result = calculate_btc_dominance_pattern(result, df_btc, timeframe)

    # 7. BTC volatility influence (6)
    result = calculate_btc_volatility_influence(result, df_btc, timeframe)

    # 8. BTC divergence signals (4)
    result = calculate_btc_divergence_signals(result, df_btc, timeframe)

    return result


def calculate_multi_tf_btc_influence(
    asset_dfs: Dict[str, pd.DataFrame],
    btc_dfs: Dict[str, pd.DataFrame],
    crypto_symbol: str
) -> Dict[str, pd.DataFrame]:
    """
    Calculate BTC influence features for all 3 timeframes

    Args:
        asset_dfs: Dict with keys '4h', '1d', '1w' (asset dataframes)
        btc_dfs: Dict with keys '4h', '1d', '1w' (BTC dataframes)
        crypto_symbol: 'BTCUSDT', 'ETHUSDT', or 'SOLUSDT'

    Returns:
        Dict with keys '4h', '1d', '1w' containing DataFrames with BTC influence features

    Feature count:
        - Per timeframe: 37 BTC influence features (ETH/SOL only)
        - Total: 111 BTC influence features (37 × 3 timeframes)
        - BTC: 0 features (no self-influence)
    """
    return {
        '4h': calculate_btc_influence_features(
            asset_dfs['4h'], btc_dfs['4h'], '4h', crypto_symbol
        ),
        '1d': calculate_btc_influence_features(
            asset_dfs['1d'], btc_dfs['1d'], '1d', crypto_symbol
        ),
        '1w': calculate_btc_influence_features(
            asset_dfs['1w'], btc_dfs['1w'], '1w', crypto_symbol
        )
    }
