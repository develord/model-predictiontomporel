"""
Create exactly 90 features matching required_features.json
"""
import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, VolumePriceTrendIndicator


def create_90_features(df):
    """Create exactly 90 features matching required_features.json"""

    df = df.copy()

    # SMA indicators and ratios
    for period in [5, 10, 20, 30, 50, 100, 200]:
        df[f'sma_{period}'] = SMAIndicator(df['close'], window=period).sma_indicator()
        df[f'close_sma_{period}_ratio'] = df['close'] / df[f'sma_{period}']

    # EMA indicators and ratios
    for period in [5, 10, 20, 50]:
        df[f'ema_{period}'] = EMAIndicator(df['close'], window=period).ema_indicator()
        df[f'close_ema_{period}_ratio'] = df['close'] / df[f'ema_{period}']

    # RSI indicators
    for period in [7, 14, 21, 28]:
        df[f'rsi_{period}'] = RSIIndicator(df['close'], window=period).rsi()

    # MACD
    ema_12 = EMAIndicator(df['close'], window=12).ema_indicator()
    ema_26 = EMAIndicator(df['close'], window=26).ema_indicator()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = EMAIndicator(df['macd'], window=9).ema_indicator()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # Bollinger Bands (20 and 50 periods)
    for period in [20, 50]:
        bb = BollingerBands(df['close'], window=period, window_dev=2)
        df[f'bb_upper_{period}'] = bb.bollinger_hband()
        df[f'bb_lower_{period}'] = bb.bollinger_lband()
        df[f'bb_middle_{period}'] = bb.bollinger_mavg()
        df[f'bb_width_{period}'] = df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']
        df[f'bb_position_{period}'] = (df['close'] - df[f'bb_lower_{period}']) / (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'])

    # ATR indicators
    for period in [7, 14, 21]:
        atr = AverageTrueRange(df['high'], df['low'], df['close'], window=period)
        df[f'atr_{period}'] = atr.average_true_range()
        df[f'atr_{period}_pct'] = df[f'atr_{period}'] / df['close'] * 100

    # Volume indicators
    for period in [5, 10, 20]:
        df[f'volume_sma_{period}'] = df['volume'].rolling(window=period).mean()
        df[f'volume_ratio_{period}'] = df['volume'] / df[f'volume_sma_{period}']

    # Volume Price Trend and OBV
    df['vpt'] = VolumePriceTrendIndicator(df['close'], df['volume']).volume_price_trend()
    df['obv'] = OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()

    # Rate of Change (ROC)
    for period in [5, 10, 20]:
        df[f'roc_{period}'] = ((df['close'] - df['close'].shift(period)) / df['close'].shift(period)) * 100

    # Momentum
    for period in [5, 10, 20]:
        df[f'momentum_{period}'] = df['close'] - df['close'].shift(period)

    # Standard Deviation (Volatility)
    for period in [5, 10, 20]:
        df[f'std_{period}'] = df['close'].rolling(window=period).std()

    # Historical Volatility
    for period in [10, 20, 30]:
        returns = np.log(df['close'] / df['close'].shift(1))
        df[f'hist_vol_{period}'] = returns.rolling(window=period).std() * np.sqrt(period)

    # Price features
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df['hl_spread'] = df['high'] - df['low']
    df['hl_ratio'] = df['high'] / df['low']
    df['oc_spread'] = df['close'] - df['open']
    df['body_size'] = abs(df['close'] - df['open'])
    df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']

    # ADX
    adx_indicator = ADXIndicator(df['high'], df['low'], df['close'], window=14)
    df['adx'] = adx_indicator.adx()
    df['plus_di'] = adx_indicator.adx_pos()
    df['minus_di'] = adx_indicator.adx_neg()

    # Statistical features
    for period in [10, 20]:
        df[f'skew_{period}'] = df['close'].rolling(window=period).skew()
        df[f'kurt_{period}'] = df['close'].rolling(window=period).kurt()
        df[f'close_rank_{period}'] = df['close'].rolling(window=period).apply(
            lambda x: (x.rank().iloc[-1] / len(x)) if len(x) > 0 else 0.5
        )

    # Candlestick patterns (simplified)
    df['doji'] = ((abs(df['close'] - df['open']) / (df['high'] - df['low'])) < 0.1).astype(int)
    df['hammer'] = (
        ((df['lower_shadow'] > 2 * df['body_size']) &
         (df['upper_shadow'] < df['body_size']))
    ).astype(int)
    df['bullish_engulfing'] = (
        ((df['close'] > df['open']) &
         (df['close'].shift(1) < df['open'].shift(1)) &
         (df['close'] > df['open'].shift(1)) &
         (df['open'] < df['close'].shift(1)))
    ).astype(int)
    df['bearish_engulfing'] = (
        ((df['close'] < df['open']) &
         (df['close'].shift(1) > df['open'].shift(1)) &
         (df['close'] < df['open'].shift(1)) &
         (df['open'] > df['close'].shift(1)))
    ).astype(int)

    # Time features
    df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek
    df['day_of_month'] = pd.to_datetime(df['date']).dt.day
    df['month'] = pd.to_datetime(df['date']).dt.month
    df['quarter'] = pd.to_datetime(df['date']).dt.quarter

    return df


def get_feature_columns():
    """Return the exact 90 feature columns in order"""
    return [
        "sma_5", "close_sma_5_ratio", "sma_10", "close_sma_10_ratio",
        "sma_20", "close_sma_20_ratio", "sma_30", "close_sma_30_ratio",
        "sma_50", "close_sma_50_ratio", "sma_100", "close_sma_100_ratio",
        "sma_200", "close_sma_200_ratio",
        "ema_5", "close_ema_5_ratio", "ema_10", "close_ema_10_ratio",
        "ema_20", "close_ema_20_ratio", "ema_50", "close_ema_50_ratio",
        "rsi_7", "rsi_14", "rsi_21", "rsi_28",
        "macd", "macd_signal", "macd_hist",
        "bb_upper_20", "bb_lower_20", "bb_middle_20", "bb_width_20", "bb_position_20",
        "bb_upper_50", "bb_lower_50", "bb_middle_50", "bb_width_50", "bb_position_50",
        "atr_7", "atr_7_pct", "atr_14", "atr_14_pct", "atr_21", "atr_21_pct",
        "volume_sma_5", "volume_ratio_5", "volume_sma_10", "volume_ratio_10",
        "volume_sma_20", "volume_ratio_20",
        "vpt", "obv",
        "roc_5", "roc_10", "roc_20",
        "momentum_5", "momentum_10", "momentum_20",
        "std_5", "std_10", "std_20",
        "hist_vol_10", "hist_vol_20", "hist_vol_30",
        "returns", "log_returns", "hl_spread", "hl_ratio", "oc_spread",
        "body_size", "upper_shadow", "lower_shadow",
        "adx", "plus_di", "minus_di",
        "skew_10", "kurt_10", "skew_20", "kurt_20",
        "close_rank_10", "close_rank_20",
        "doji", "hammer", "bullish_engulfing", "bearish_engulfing",
        "day_of_week", "day_of_month", "month", "quarter"
    ]


if __name__ == "__main__":
    # Test
    print("Testing 90 features generation...")

    # Load sample data
    import sys
    from pathlib import Path
    BASE_DIR = Path(__file__).parent.parent

    df = pd.read_csv(BASE_DIR / 'data' / 'cache' / 'btc_1d_data.csv')
    print(f"Loaded {len(df)} rows of data")

    # Create features
    df_features = create_90_features(df)

    # Get feature columns
    feature_cols = get_feature_columns()
    print(f"\nExpected 90 features, got {len(feature_cols)} feature names")

    # Check all features exist
    missing = [f for f in feature_cols if f not in df_features.columns]
    if missing:
        print(f"\nMissing features: {missing}")
    else:
        print("\n✓ All 90 features present!")

    # Show sample
    print(f"\nSample features (first 5):")
    print(df_features[feature_cols[:5]].head())

    print(f"\n✓ Feature generation successful!")
    print(f"Total columns: {len(df_features.columns)}")
    print(f"Feature columns: {len([c for c in df_features.columns if c in feature_cols])}")
