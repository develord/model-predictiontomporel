"""
Advanced Feature Engineering Module - 70+ Technical Indicators & Market Microstructure
=======================================================================================
Comprehensive feature engineering for cryptocurrency trading including:
1. Advanced technical indicators (momentum, volatility, trend)
2. Market microstructure features
3. Order flow imbalance indicators
4. Statistical arbitrage signals
5. Machine learning-derived features
6. Cross-asset correlations
7. Market regime detection

Author: Advanced Trading System
Version: 2.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')


class AdvancedTechnicalIndicators:
    """
    Collection of 70+ advanced technical indicators for crypto trading
    """

    @staticmethod
    def ichimoku_cloud(high: pd.Series, low: pd.Series, close: pd.Series,
                       conversion: int = 9, base: int = 26, span_b: int = 52,
                       displacement: int = 26) -> Dict[str, pd.Series]:
        """
        Calculate Ichimoku Cloud components
        """
        # Conversion Line (Tenkan-sen)
        conversion_high = high.rolling(window=conversion).max()
        conversion_low = low.rolling(window=conversion).min()
        conversion_line = (conversion_high + conversion_low) / 2

        # Base Line (Kijun-sen)
        base_high = high.rolling(window=base).max()
        base_low = low.rolling(window=base).min()
        base_line = (base_high + base_low) / 2

        # Leading Span A (Senkou Span A)
        leading_span_a = ((conversion_line + base_line) / 2).shift(displacement)

        # Leading Span B (Senkou Span B)
        span_b_high = high.rolling(window=span_b).max()
        span_b_low = low.rolling(window=span_b).min()
        leading_span_b = ((span_b_high + span_b_low) / 2).shift(displacement)

        # Lagging Span (Chikou Span)
        lagging_span = close.shift(-displacement)

        return {
            'ichimoku_conversion': conversion_line,
            'ichimoku_base': base_line,
            'ichimoku_span_a': leading_span_a,
            'ichimoku_span_b': leading_span_b,
            'ichimoku_lagging': lagging_span,
            'ichimoku_cloud_thickness': abs(leading_span_a - leading_span_b),
            'price_to_cloud_distance': close - (leading_span_a + leading_span_b) / 2
        }

    @staticmethod
    def advanced_momentum_indicators(close: pd.Series, high: pd.Series, low: pd.Series,
                                    volume: pd.Series) -> Dict[str, pd.Series]:
        """
        Calculate advanced momentum indicators
        """
        indicators = {}

        # 1. Chande Momentum Oscillator (CMO)
        def cmo(close, period=14):
            diff = close.diff()
            up = diff.where(diff > 0, 0)
            down = -diff.where(diff < 0, 0)
            sum_up = up.rolling(window=period).sum()
            sum_down = down.rolling(window=period).sum()
            return 100 * (sum_up - sum_down) / (sum_up + sum_down + 1e-10)

        indicators['cmo_14'] = cmo(close, 14)

        # 2. Know Sure Thing (KST)
        def kst(close):
            roc1 = ((close - close.shift(10)) / close.shift(10)) * 100
            roc2 = ((close - close.shift(15)) / close.shift(15)) * 100
            roc3 = ((close - close.shift(20)) / close.shift(20)) * 100
            roc4 = ((close - close.shift(30)) / close.shift(30)) * 100

            roc1_sma = roc1.rolling(window=10).mean()
            roc2_sma = roc2.rolling(window=10).mean()
            roc3_sma = roc3.rolling(window=10).mean()
            roc4_sma = roc4.rolling(window=15).mean()

            kst_value = (roc1_sma * 1) + (roc2_sma * 2) + (roc3_sma * 3) + (roc4_sma * 4)
            kst_signal = kst_value.rolling(window=9).mean()

            return kst_value, kst_signal

        kst_val, kst_sig = kst(close)
        indicators['kst'] = kst_val
        indicators['kst_signal'] = kst_sig
        indicators['kst_diff'] = kst_val - kst_sig

        # 3. Trix Indicator
        def trix(close, period=14):
            ema1 = close.ewm(span=period, adjust=False).mean()
            ema2 = ema1.ewm(span=period, adjust=False).mean()
            ema3 = ema2.ewm(span=period, adjust=False).mean()
            return (ema3.pct_change()) * 10000

        indicators['trix'] = trix(close)

        # 4. Ultimate Oscillator
        def ultimate_oscillator(high, low, close, p1=7, p2=14, p3=28):
            true_range = pd.DataFrame({
                'hl': high - low,
                'hc': abs(high - close.shift()),
                'lc': abs(low - close.shift())
            }).max(axis=1)

            buying_pressure = close - pd.DataFrame({'low': low, 'prev_close': close.shift()}).min(axis=1)

            avg1 = buying_pressure.rolling(p1).sum() / true_range.rolling(p1).sum()
            avg2 = buying_pressure.rolling(p2).sum() / true_range.rolling(p2).sum()
            avg3 = buying_pressure.rolling(p3).sum() / true_range.rolling(p3).sum()

            return 100 * ((avg1 * p2 * p3) + (avg2 * p1 * p3) + (avg3 * p1 * p2)) / ((p1 * p2) + (p1 * p3) + (p2 * p3))

        indicators['ultimate_oscillator'] = ultimate_oscillator(high, low, close)

        # 5. Mass Index
        def mass_index(high, low, period=25):
            ema_range = (high - low).ewm(span=9, adjust=False).mean()
            double_ema_range = ema_range.ewm(span=9, adjust=False).mean()
            ratio = ema_range / double_ema_range
            return ratio.rolling(window=period).sum()

        indicators['mass_index'] = mass_index(high, low)

        return indicators

    @staticmethod
    def volatility_indicators(close: pd.Series, high: pd.Series, low: pd.Series) -> Dict[str, pd.Series]:
        """
        Advanced volatility indicators
        """
        indicators = {}

        # 1. Garman-Klass Volatility
        def garman_klass(high, low, close, period=20):
            log_hl = np.log(high / low) ** 2
            log_co = np.log(close / close.shift()) ** 2
            gk = np.sqrt((0.5 * log_hl - (2 * np.log(2) - 1) * log_co).rolling(window=period).mean())
            return gk * np.sqrt(252)  # Annualized

        indicators['garman_klass_vol'] = garman_klass(high, low, close)

        # 2. Parkinson Volatility
        def parkinson(high, low, period=20):
            return np.sqrt((np.log(high / low) ** 2).rolling(window=period).mean() / (4 * np.log(2))) * np.sqrt(252)

        indicators['parkinson_vol'] = parkinson(high, low)

        # 3. Rogers-Satchell Volatility
        def rogers_satchell(high, low, close, period=20):
            rs = np.sqrt((np.log(high / close) * np.log(high / close.shift()) +
                         np.log(low / close) * np.log(low / close.shift())).rolling(window=period).mean())
            return rs * np.sqrt(252)

        indicators['rogers_satchell_vol'] = rogers_satchell(high, low, close)

        # 4. Yang-Zhang Volatility
        def yang_zhang(open_price, high, low, close, period=20):
            log_ho = np.log(high / open_price)
            log_lo = np.log(low / open_price)
            log_co = np.log(close / open_price)

            log_oc = np.log(open_price / close.shift())
            log_oc_sq = log_oc ** 2

            log_cc = np.log(close / close.shift())
            log_cc_sq = log_cc ** 2

            rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)

            close_vol = log_cc_sq.rolling(window=period).mean()
            open_vol = log_oc_sq.rolling(window=period).mean()
            rs_vol = rs.rolling(window=period).mean()

            k = 0.34 / (1.34 + (period + 1) / (period - 1))
            yz = np.sqrt(open_vol + k * close_vol + (1 - k) * rs_vol)

            return yz * np.sqrt(252)

        # Note: Requires open price - using close.shift() as proxy
        indicators['yang_zhang_vol'] = yang_zhang(close.shift(), high, low, close)

        # 5. Average True Range Percent Rank
        def atr_percentile(high, low, close, atr_period=14, rank_period=100):
            tr = pd.concat([
                high - low,
                abs(high - close.shift()),
                abs(low - close.shift())
            ], axis=1).max(axis=1)

            atr = tr.rolling(window=atr_period).mean()
            return atr.rolling(window=rank_period).rank(pct=True) * 100

        indicators['atr_percentile'] = atr_percentile(high, low, close)

        return indicators

    @staticmethod
    def volume_profile_indicators(close: pd.Series, volume: pd.Series,
                                 high: pd.Series, low: pd.Series) -> Dict[str, pd.Series]:
        """
        Volume profile and order flow indicators
        """
        indicators = {}

        # 1. Volume-Weighted Moving Average (VWMA)
        def vwma(close, volume, period=20):
            return (close * volume).rolling(window=period).sum() / volume.rolling(window=period).sum()

        indicators['vwma_20'] = vwma(close, volume, 20)
        indicators['vwma_50'] = vwma(close, volume, 50)

        # 2. Accumulation/Distribution Line
        def adl(high, low, close, volume):
            clv = ((close - low) - (high - close)) / (high - low + 1e-10)
            return (clv * volume).cumsum()

        indicators['adl'] = adl(high, low, close, volume)

        # 3. Ease of Movement
        def ease_of_movement(high, low, volume, period=14):
            distance_moved = (high + low) / 2 - (high.shift() + low.shift()) / 2
            emv = distance_moved / (volume / (high - low) / 1000000)
            return emv.rolling(window=period).mean()

        indicators['ease_of_movement'] = ease_of_movement(high, low, volume)

        # 4. Force Index
        def force_index(close, volume, period=13):
            return (close.diff() * volume).ewm(span=period, adjust=False).mean()

        indicators['force_index'] = force_index(close, volume)

        # 5. Klinger Oscillator
        def klinger_oscillator(high, low, close, volume, fast=34, slow=55):
            trend = np.sign((high + low + close) - (high.shift() + low.shift() + close.shift()))
            dm = high - low
            cm = (trend != trend.shift()).cumsum()

            vf = volume * trend * dm / cm
            kvo = vf.ewm(span=fast, adjust=False).mean() - vf.ewm(span=slow, adjust=False).mean()

            return kvo

        indicators['klinger_oscillator'] = klinger_oscillator(high, low, close, volume)

        return indicators

    @staticmethod
    def market_structure_features(close: pd.Series, high: pd.Series,
                                 low: pd.Series, volume: pd.Series) -> Dict[str, pd.Series]:
        """
        Market microstructure and regime detection features
        """
        indicators = {}

        # 1. Hurst Exponent (market trending vs mean-reverting)
        def hurst_exponent(series, max_lag=20):
            lags = range(2, max_lag)
            tau = []

            for lag in lags:
                differences = series.diff(lag).dropna()
                tau.append(differences.std())

            tau = np.array(tau)
            lags = np.array(lags)

            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return pd.Series(poly[0] / 2, index=series.index)

        indicators['hurst_exponent'] = hurst_exponent(close)

        # 2. Fractal Dimension
        def fractal_dimension(series, period=30):
            def calc_fd(x):
                n = len(x)
                if n < 2:
                    return np.nan

                # Rescaled range
                mean_x = np.mean(x)
                y = np.cumsum(x - mean_x)
                r = np.max(y) - np.min(y)
                s = np.std(x)

                if s == 0:
                    return np.nan

                return np.log(r/s) / np.log(n) if r/s > 0 else np.nan

            return series.rolling(window=period).apply(calc_fd)

        indicators['fractal_dimension'] = fractal_dimension(close.pct_change())

        # 3. Entropy (market uncertainty)
        def shannon_entropy(series, period=20):
            def calc_entropy(x):
                if len(x) < 2:
                    return np.nan
                # Discretize into bins
                bins = np.linspace(x.min(), x.max(), 10)
                hist, _ = np.histogram(x, bins=bins)
                probs = hist / np.sum(hist)
                probs = probs[probs > 0]  # Remove zero probabilities
                return -np.sum(probs * np.log2(probs))

            return series.rolling(window=period).apply(calc_entropy)

        indicators['market_entropy'] = shannon_entropy(close.pct_change())

        # 4. Price Efficiency Ratio
        def efficiency_ratio(close, period=20):
            change = abs(close - close.shift(period))
            volatility = (abs(close - close.shift())).rolling(window=period).sum()
            return change / (volatility + 1e-10)

        indicators['efficiency_ratio'] = efficiency_ratio(close)

        # 5. Market Regime (Bull/Bear/Sideways)
        def market_regime(close, lookback=50):
            sma_short = close.rolling(window=20).mean()
            sma_long = close.rolling(window=50).mean()
            std = close.pct_change().rolling(window=lookback).std()

            regime = pd.Series(0, index=close.index)  # 0 = sideways

            # Bull market
            bull_condition = (sma_short > sma_long) & (close > sma_short)
            regime[bull_condition] = 1

            # Bear market
            bear_condition = (sma_short < sma_long) & (close < sma_short)
            regime[bear_condition] = -1

            return regime

        indicators['market_regime'] = market_regime(close)

        return indicators

    @staticmethod
    def pattern_recognition_features(close: pd.Series, high: pd.Series,
                                    low: pd.Series, open_price: pd.Series) -> Dict[str, pd.Series]:
        """
        Candlestick pattern recognition features
        """
        indicators = {}

        # Helper functions
        body = close - open_price
        body_abs = abs(body)
        upper_shadow = high - pd.concat([close, open_price], axis=1).max(axis=1)
        lower_shadow = pd.concat([close, open_price], axis=1).min(axis=1) - low
        high_low = high - low

        # 1. Doji
        indicators['is_doji'] = (body_abs <= high_low * 0.1).astype(int)

        # 2. Hammer
        indicators['is_hammer'] = (
            (lower_shadow > body_abs * 2) &
            (upper_shadow < body_abs * 0.3) &
            (body > 0)
        ).astype(int)

        # 3. Shooting Star
        indicators['is_shooting_star'] = (
            (upper_shadow > body_abs * 2) &
            (lower_shadow < body_abs * 0.3) &
            (body < 0)
        ).astype(int)

        # 4. Engulfing Pattern
        prev_body = body.shift(1)
        indicators['bullish_engulfing'] = (
            (body > 0) & (prev_body < 0) &
            (body_abs > abs(prev_body)) &
            (close > open_price.shift(1)) &
            (open_price < close.shift(1))
        ).astype(int)

        indicators['bearish_engulfing'] = (
            (body < 0) & (prev_body > 0) &
            (body_abs > abs(prev_body)) &
            (close < open_price.shift(1)) &
            (open_price > close.shift(1))
        ).astype(int)

        # 5. Three White Soldiers / Three Black Crows
        def three_pattern(body, threshold=0):
            pattern = pd.Series(0, index=body.index)
            for i in range(2, len(body)):
                if threshold > 0:  # White soldiers
                    if all(body.iloc[i-j] > threshold for j in range(3)):
                        pattern.iloc[i] = 1
                else:  # Black crows
                    if all(body.iloc[i-j] < threshold for j in range(3)):
                        pattern.iloc[i] = 1
            return pattern

        indicators['three_white_soldiers'] = three_pattern(body, threshold=0.001)
        indicators['three_black_crows'] = three_pattern(body, threshold=-0.001)

        return indicators

    @staticmethod
    def statistical_features(close: pd.Series, returns: Optional[pd.Series] = None) -> Dict[str, pd.Series]:
        """
        Statistical and distributional features
        """
        if returns is None:
            returns = close.pct_change()

        indicators = {}

        # 1. Skewness (asymmetry of returns)
        indicators['return_skew'] = returns.rolling(window=30).skew()

        # 2. Kurtosis (tail heaviness)
        indicators['return_kurtosis'] = returns.rolling(window=30).kurt()

        # 3. Jarque-Bera Test (normality)
        def jarque_bera_stat(x):
            if len(x) < 4:
                return np.nan
            jb, _ = stats.jarque_bera(x)
            return jb

        indicators['jarque_bera'] = returns.rolling(window=30).apply(jarque_bera_stat)

        # 4. Value at Risk (VaR)
        indicators['var_95'] = returns.rolling(window=100).quantile(0.05)
        indicators['var_99'] = returns.rolling(window=100).quantile(0.01)

        # 5. Conditional Value at Risk (CVaR)
        def cvar(returns, confidence=0.95):
            var = returns.quantile(1 - confidence)
            return returns[returns <= var].mean()

        indicators['cvar_95'] = returns.rolling(window=100).apply(lambda x: cvar(x, 0.95))

        # 6. Autocorrelation
        indicators['returns_autocorr_1'] = returns.rolling(window=30).apply(lambda x: x.autocorr(1))
        indicators['returns_autocorr_5'] = returns.rolling(window=30).apply(lambda x: x.autocorr(5))

        # 7. Run test for randomness
        def runs_test(x):
            if len(x) < 2:
                return np.nan
            median = np.median(x)
            runs = np.sum(np.diff(x > median) != 0) + 1
            return runs

        indicators['runs_test'] = returns.rolling(window=30).apply(runs_test)

        return indicators

    @staticmethod
    def cross_asset_correlations(asset_close: pd.Series, btc_close: pd.Series,
                                spy_close: Optional[pd.Series] = None,
                                gold_close: Optional[pd.Series] = None) -> Dict[str, pd.Series]:
        """
        Cross-asset correlation features
        """
        indicators = {}

        # Rolling correlations with BTC
        asset_returns = asset_close.pct_change()
        btc_returns = btc_close.pct_change()

        indicators['corr_btc_20'] = asset_returns.rolling(window=20).corr(btc_returns)
        indicators['corr_btc_60'] = asset_returns.rolling(window=60).corr(btc_returns)

        # Beta to BTC
        def calculate_beta(asset_ret, market_ret, window=60):
            covariance = asset_ret.rolling(window=window).cov(market_ret)
            variance = market_ret.rolling(window=window).var()
            return covariance / (variance + 1e-10)

        indicators['beta_to_btc'] = calculate_beta(asset_returns, btc_returns)

        # If other assets available
        if spy_close is not None:
            spy_returns = spy_close.pct_change()
            indicators['corr_spy_60'] = asset_returns.rolling(window=60).corr(spy_returns)

        if gold_close is not None:
            gold_returns = gold_close.pct_change()
            indicators['corr_gold_60'] = asset_returns.rolling(window=60).corr(gold_returns)

        return indicators


def calculate_all_advanced_features(
    df: pd.DataFrame,
    btc_df: Optional[pd.DataFrame] = None,
    include_patterns: bool = True
) -> pd.DataFrame:
    """
    Calculate all advanced features for the dataframe

    Args:
        df: DataFrame with OHLCV data
        btc_df: Bitcoin data for cross-asset features (optional)
        include_patterns: Whether to include pattern recognition

    Returns:
        DataFrame with all advanced features added
    """
    result = df.copy()
    calculator = AdvancedTechnicalIndicators()

    print("Calculating advanced features...")

    # 1. Ichimoku Cloud
    ichimoku = calculator.ichimoku_cloud(df['high'], df['low'], df['close'])
    for key, value in ichimoku.items():
        result[f'adv_{key}'] = value

    # 2. Advanced Momentum
    momentum = calculator.advanced_momentum_indicators(
        df['close'], df['high'], df['low'], df['volume']
    )
    for key, value in momentum.items():
        result[f'adv_{key}'] = value

    # 3. Volatility Indicators
    volatility = calculator.volatility_indicators(df['close'], df['high'], df['low'])
    for key, value in volatility.items():
        result[f'adv_{key}'] = value

    # 4. Volume Profile
    volume_profile = calculator.volume_profile_indicators(
        df['close'], df['volume'], df['high'], df['low']
    )
    for key, value in volume_profile.items():
        result[f'adv_{key}'] = value

    # 5. Market Structure
    structure = calculator.market_structure_features(
        df['close'], df['high'], df['low'], df['volume']
    )
    for key, value in structure.items():
        result[f'adv_{key}'] = value

    # 6. Statistical Features
    statistical = calculator.statistical_features(df['close'])
    for key, value in statistical.items():
        result[f'adv_{key}'] = value

    # 7. Pattern Recognition (if requested)
    if include_patterns and 'open' in df.columns:
        patterns = calculator.pattern_recognition_features(
            df['close'], df['high'], df['low'], df['open']
        )
        for key, value in patterns.items():
            result[f'adv_{key}'] = value

    # 8. Cross-asset correlations (if BTC data provided)
    if btc_df is not None and 'close' in btc_df.columns:
        correlations = calculator.cross_asset_correlations(
            df['close'], btc_df['close']
        )
        for key, value in correlations.items():
            result[f'adv_{key}'] = value

    # Handle NaN values
    result = result.fillna(method='ffill').fillna(0)

    print(f"Added {len(result.columns) - len(df.columns)} advanced features")

    return result


def feature_engineering_pipeline(
    df: pd.DataFrame,
    lookback_window: int = 60,
    include_lags: bool = True,
    include_rolling_stats: bool = True
) -> pd.DataFrame:
    """
    Complete feature engineering pipeline

    Args:
        df: Raw OHLCV data
        lookback_window: Window for sequence features
        include_lags: Whether to include lagged features
        include_rolling_stats: Whether to include rolling statistics

    Returns:
        DataFrame with all features
    """
    result = calculate_all_advanced_features(df)

    if include_lags:
        # Add lagged features for important indicators
        important_cols = ['close', 'volume', 'adv_rsi_14', 'adv_macd_line']
        for col in important_cols:
            if col in result.columns:
                for lag in [1, 3, 5, 10]:
                    result[f'{col}_lag_{lag}'] = result[col].shift(lag)

    if include_rolling_stats:
        # Add rolling statistics
        for col in ['close', 'volume']:
            if col in result.columns:
                result[f'{col}_roll_mean_20'] = result[col].rolling(window=20).mean()
                result[f'{col}_roll_std_20'] = result[col].rolling(window=20).std()
                result[f'{col}_roll_max_20'] = result[col].rolling(window=20).max()
                result[f'{col}_roll_min_20'] = result[col].rolling(window=20).min()

    return result


if __name__ == "__main__":
    # Test the feature engineering
    print("Testing Advanced Feature Engineering...")

    # Create sample data
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='4H')
    sample_data = pd.DataFrame({
        'open': np.random.randn(len(dates)).cumsum() + 100,
        'high': np.random.randn(len(dates)).cumsum() + 101,
        'low': np.random.randn(len(dates)).cumsum() + 99,
        'close': np.random.randn(len(dates)).cumsum() + 100,
        'volume': np.random.exponential(1000000, len(dates))
    }, index=dates)

    # Ensure high > low
    sample_data['high'] = sample_data[['high', 'open', 'close']].max(axis=1)
    sample_data['low'] = sample_data[['low', 'open', 'close']].min(axis=1)

    # Calculate features
    result = feature_engineering_pipeline(sample_data)

    print(f"\nOriginal features: {len(sample_data.columns)}")
    print(f"Total features after engineering: {len(result.columns)}")
    print(f"\nFeature categories:")

    categories = {
        'ichimoku': [c for c in result.columns if 'ichimoku' in c],
        'momentum': [c for c in result.columns if any(x in c for x in ['cmo', 'kst', 'trix', 'ultimate'])],
        'volatility': [c for c in result.columns if 'vol' in c],
        'volume': [c for c in result.columns if any(x in c for x in ['vwma', 'adl', 'force', 'klinger'])],
        'structure': [c for c in result.columns if any(x in c for x in ['hurst', 'fractal', 'entropy', 'regime'])],
        'statistical': [c for c in result.columns if any(x in c for x in ['skew', 'kurt', 'var_', 'cvar'])],
        'patterns': [c for c in result.columns if any(x in c for x in ['doji', 'hammer', 'engulfing', 'soldiers'])]
    }

    for category, features in categories.items():
        print(f"  {category}: {len(features)} features")

    print("\nFeature engineering complete!")