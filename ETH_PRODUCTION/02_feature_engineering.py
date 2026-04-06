"""
ETH Feature Engineering Script - Multi-Timeframe + Non-Technical (V3)
=====================================================================
Matches BTC V3 pipeline exactly:
- Multi-TF technical indicators (4h, 1d, 1w)
- Cross-timeframe alignment signals
- Non-technical features (volatility regimes, momentum coherence)
- Market regime features (28 features)
- ATR-based labeling (symmetric 1.5x ATR)

Usage:
    python 02_feature_engineering.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
import ta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data' / 'cache'

CRYPTO = 'ETH'
# ATR-based labeling params (symmetric = no directional bias)
ATR_TP_MULT = 1.5
ATR_SL_MULT = 1.5
FIXED_TP_PCT = 0.012
FIXED_SL_PCT = 0.012
BASE_LOOKAHEAD = 10
ATR_LOOKAHEAD_MULT = 0.7


def create_technical_indicators(df, prefix=''):
    """Create standard technical indicators for a timeframe"""
    df[f'{prefix}rsi_14'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    df[f'{prefix}rsi_21'] = ta.momentum.RSIIndicator(df['close'], window=21).rsi()

    macd = ta.trend.MACD(df['close'])
    df[f'{prefix}macd_line'] = macd.macd()
    df[f'{prefix}macd_signal'] = macd.macd_signal()
    df[f'{prefix}macd_histogram'] = macd.macd_diff()

    bb = ta.volatility.BollingerBands(df['close'])
    df[f'{prefix}bb_upper'] = bb.bollinger_hband()
    df[f'{prefix}bb_middle'] = bb.bollinger_mavg()
    df[f'{prefix}bb_lower'] = bb.bollinger_lband()
    df[f'{prefix}bb_width'] = bb.bollinger_wband()

    df[f'{prefix}ema_12'] = ta.trend.EMAIndicator(df['close'], window=12).ema_indicator()
    df[f'{prefix}ema_26'] = ta.trend.EMAIndicator(df['close'], window=26).ema_indicator()
    df[f'{prefix}ema_50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()

    df[f'{prefix}atr_14'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()

    stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
    df[f'{prefix}stoch_k'] = stoch.stoch()
    df[f'{prefix}stoch_d'] = stoch.stoch_signal()

    df[f'{prefix}adx_14'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()

    df[f'{prefix}obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
    df[f'{prefix}cmf_20'] = ta.volume.ChaikinMoneyFlowIndicator(df['high'], df['low'], df['close'], df['volume'], window=20).chaikin_money_flow()

    df[f'{prefix}momentum_5'] = df['close'].pct_change(5)
    df[f'{prefix}momentum_10'] = df['close'].pct_change(10)
    df[f'{prefix}hist_vol_20'] = df['close'].pct_change().rolling(20).std()
    df[f'{prefix}volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()

    return df


def create_cross_tf_features(df):
    """Create cross-timeframe alignment and divergence features"""
    logger.info("  Creating cross-timeframe features...")

    for tf_pair in [('1d', '4h'), ('1d', '1w'), ('4h', '1w')]:
        tf1, tf2 = tf_pair
        col1 = f'{tf1}_rsi_14'
        col2 = f'{tf2}_rsi_14'
        if col1 in df.columns and col2 in df.columns:
            df[f'rsi_diff_{tf1}_{tf2}'] = df[col1] - df[col2]

    rsi_cols = [c for c in df.columns if c.endswith('_rsi_14')]
    if len(rsi_cols) >= 2:
        df['rsi_bullish_count'] = sum((df[c] > 50).astype(int) for c in rsi_cols)
        df['rsi_oversold_count'] = sum((df[c] < 30).astype(int) for c in rsi_cols)
        df['rsi_overbought_count'] = sum((df[c] > 70).astype(int) for c in rsi_cols)

    macd_cols = [c for c in df.columns if c.endswith('_macd_histogram')]
    if len(macd_cols) >= 2:
        df['macd_bullish_count'] = sum((df[c] > 0).astype(int) for c in macd_cols)

    mom_cols = [c for c in df.columns if c.endswith('_momentum_5')]
    if len(mom_cols) >= 2:
        df['momentum_bullish_count'] = sum((df[c] > 0).astype(int) for c in mom_cols)

    adx_cols = [c for c in df.columns if c.endswith('_adx_14')]
    if len(adx_cols) >= 2:
        df['adx_strong_count'] = sum((df[c] > 25).astype(int) for c in adx_cols)
        df['adx_mean'] = sum(df[c] for c in adx_cols) / len(adx_cols)

    vol_cols = [c for c in df.columns if c.endswith('_hist_vol_20')]
    if len(vol_cols) >= 2:
        df['vol_mean_all_tf'] = sum(df[c] for c in vol_cols) / len(vol_cols)

    return df


def create_non_technical_features(df):
    """Create non-technical / market structure features"""
    logger.info("  Creating non-technical features...")

    df['daily_range_pct'] = (df['high'] - df['low']) / df['close']
    df['daily_range_ma5'] = df['daily_range_pct'].rolling(5).mean()
    df['daily_range_ma20'] = df['daily_range_pct'].rolling(20).mean()
    df['volatility_regime'] = (df['daily_range_ma5'] / df['daily_range_ma20']).fillna(1)

    df['volume_ma5'] = df['volume'].rolling(5).mean()
    df['volume_ma20'] = df['volume'].rolling(20).mean()
    df['volume_trend'] = (df['volume_ma5'] / df['volume_ma20']).fillna(1)

    df['price_position_20'] = (df['close'] - df['low'].rolling(20).min()) / \
                               (df['high'].rolling(20).max() - df['low'].rolling(20).min() + 1e-10)
    df['price_position_50'] = (df['close'] - df['low'].rolling(50).min()) / \
                               (df['high'].rolling(50).max() - df['low'].rolling(50).min() + 1e-10)

    body = abs(df['close'] - df['open'])
    wick = df['high'] - df['low']
    df['body_ratio'] = body / (wick + 1e-10)
    df['upper_shadow_ratio'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (wick + 1e-10)
    df['lower_shadow_ratio'] = (df[['open', 'close']].min(axis=1) - df['low']) / (wick + 1e-10)

    df['returns'] = df['close'].pct_change()
    df['consecutive_up'] = 0
    df['consecutive_down'] = 0
    for i in range(1, len(df)):
        if df.iloc[i]['returns'] > 0:
            df.iloc[i, df.columns.get_loc('consecutive_up')] = df.iloc[i-1]['consecutive_up'] + 1
        if df.iloc[i]['returns'] < 0:
            df.iloc[i, df.columns.get_loc('consecutive_down')] = df.iloc[i-1]['consecutive_down'] + 1

    df['distance_from_sma20'] = (df['close'] / df['close'].rolling(20).mean() - 1)
    df['distance_from_sma50'] = (df['close'] / df['close'].rolling(50).mean() - 1)

    df['higher_highs'] = (df['high'] > df['high'].shift(1)).rolling(5).sum()
    df['lower_lows'] = (df['low'] < df['low'].shift(1)).rolling(5).sum()
    df['trend_score'] = df['higher_highs'] - df['lower_lows']

    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_month'] = df['date'].dt.day
    df['month'] = df['date'].dt.month

    return df


def create_market_regime_features(df):
    """Create market regime features: bull/bear/range detection, support/resistance, phase"""
    logger.info("  Creating market regime features...")

    sma_50 = df['close'].rolling(50).mean()
    sma_200 = df['close'].rolling(200).mean()
    df['sma_50'] = sma_50
    df['sma_200'] = sma_200
    df['sma50_above_sma200'] = (sma_50 > sma_200).astype(int)
    df['sma_spread_pct'] = (sma_50 / sma_200 - 1) * 100

    ema_20 = df['close'].rolling(20).mean()
    slope_20 = ema_20.pct_change(5) * 100
    df['regime_slope'] = slope_20
    df['regime_bull'] = ((slope_20 > 0.5) & (df['close'] > sma_50)).astype(int)
    df['regime_bear'] = ((slope_20 < -0.5) & (df['close'] < sma_50)).astype(int)
    df['regime_range'] = ((slope_20.abs() <= 0.5)).astype(int)

    high_20 = df['high'].rolling(20).max()
    low_20 = df['low'].rolling(20).min()
    df['resistance_dist_pct'] = (high_20 / df['close'] - 1) * 100
    df['support_dist_pct'] = (1 - low_20 / df['close']) * 100
    df['sr_range_pct'] = (high_20 - low_20) / df['close'] * 100

    high_50 = df['high'].rolling(50).max()
    low_50 = df['low'].rolling(50).min()
    df['resistance_50_dist_pct'] = (high_50 / df['close'] - 1) * 100
    df['support_50_dist_pct'] = (1 - low_50 / df['close']) * 100

    vol_trend = df['volume'].rolling(10).mean() / df['volume'].rolling(30).mean()
    price_pos = (df['close'] - low_20) / (high_20 - low_20 + 1e-10)
    df['volume_trend_ratio'] = vol_trend
    df['price_position_in_range'] = price_pos

    df['accumulation_score'] = (
        (df['regime_range'] == 1).astype(float) *
        (vol_trend > 1).astype(float) *
        (price_pos < 0.3).astype(float)
    )
    df['distribution_score'] = (
        (df['regime_range'] == 1).astype(float) *
        (vol_trend > 1).astype(float) *
        (price_pos > 0.7).astype(float)
    )

    sma_cross = (sma_50 > sma_200).astype(int).diff()
    df['golden_cross_5d'] = sma_cross.rolling(5).sum().clip(0, 1)
    df['death_cross_5d'] = (-sma_cross).rolling(5).sum().clip(0, 1)

    rsi = df['1d_rsi_14'] if '1d_rsi_14' in df.columns else ta.momentum.RSIIndicator(df['close'], 14).rsi()
    df['rsi_regime_shift'] = rsi.diff(5)

    returns = df['close'].pct_change()
    vol_weighted_return = (returns * df['volume']).rolling(10).sum() / df['volume'].rolling(10).sum()
    df['vwap_trend_10'] = vol_weighted_return * 100

    df['buying_pressure'] = ((df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10))
    df['selling_pressure'] = ((df['high'] - df['close']) / (df['high'] - df['low'] + 1e-10))
    df['pressure_ratio'] = df['buying_pressure'] / (df['selling_pressure'] + 1e-10)

    df['trend_consistency_10'] = returns.rolling(10).apply(lambda x: (x > 0).sum() / len(x), raw=True)
    df['trend_consistency_20'] = returns.rolling(20).apply(lambda x: (x > 0).sum() / len(x), raw=True)

    return df


def create_btc_influence_features(df):
    """Create BTC influence features for ETH — BTC leads the market"""
    logger.info("  Creating BTC influence features...")

    # Load BTC 1d data
    btc_file = DATA_DIR.parent.parent.parent / 'BTC_PRODUCTION' / 'data' / 'cache' / 'btc_1d_data.csv'
    if not btc_file.exists():
        # Try downloading
        logger.warning(f"  BTC data not found at {btc_file}, trying alt path...")
        btc_file = DATA_DIR / 'btc_1d_data.csv'
    if not btc_file.exists():
        logger.warning("  BTC data not found, downloading...")
        import ccxt
        ex = ccxt.binance({'enableRateLimit': True})
        since = int(pd.Timestamp('2017-01-01').timestamp() * 1000)
        candles = []
        while True:
            c = ex.fetch_ohlcv('BTC/USDT', '1d', since=since, limit=1000)
            if not c:
                break
            candles.extend(c)
            since = c[-1][0] + 1
            if len(c) < 1000:
                break
        btc_df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        btc_df['date'] = pd.to_datetime(btc_df['timestamp'], unit='ms')
        btc_df.to_csv(btc_file, index=False)
    else:
        btc_df = pd.read_csv(btc_file)
        btc_df['date'] = pd.to_datetime(btc_df['date'])

    # Merge BTC close/volume into ETH df
    btc_cols = btc_df[['date', 'close', 'volume', 'high', 'low']].rename(
        columns={'close': 'btc_close', 'volume': 'btc_volume', 'high': 'btc_high', 'low': 'btc_low'}
    )
    df = pd.merge_asof(df.sort_values('date'), btc_cols.sort_values('date'), on='date', direction='backward')

    # --- ETH/BTC ratio ---
    df['eth_btc_ratio'] = df['close'] / (df['btc_close'] + 1e-10)
    df['eth_btc_ratio_sma20'] = df['eth_btc_ratio'].rolling(20).mean()
    df['eth_btc_ratio_position'] = (df['eth_btc_ratio'] / df['eth_btc_ratio_sma20'] - 1) * 100
    df['eth_btc_ratio_trend'] = df['eth_btc_ratio'].pct_change(5) * 100  # 5-day trend

    # --- Correlation ---
    eth_ret = df['close'].pct_change()
    btc_ret = df['btc_close'].pct_change()
    for w in [7, 14, 30]:
        df[f'eth_btc_corr_{w}'] = eth_ret.rolling(w).corr(btc_ret)

    # --- BTC momentum ---
    for p in [5, 10, 20]:
        df[f'btc_momentum_{p}'] = df['btc_close'].pct_change(p) * 100
        df[f'eth_momentum_{p}'] = df['close'].pct_change(p) * 100

    # --- Momentum divergence: ETH outperforms or underperforms BTC ---
    for p in [5, 10]:
        df[f'eth_btc_mom_diff_{p}'] = df[f'eth_momentum_{p}'] - df[f'btc_momentum_{p}']

    # --- BTC trend (SMA) ---
    btc_sma20 = df['btc_close'].rolling(20).mean()
    btc_sma50 = df['btc_close'].rolling(50).mean()
    df['btc_above_sma20'] = (df['btc_close'] > btc_sma20).astype(int)
    df['btc_above_sma50'] = (df['btc_close'] > btc_sma50).astype(int)
    df['btc_sma20_dist'] = (df['btc_close'] / btc_sma20 - 1) * 100
    df['btc_sma50_dist'] = (df['btc_close'] / btc_sma50 - 1) * 100

    # --- BTC regime (bull/bear) ---
    btc_slope = btc_sma20.pct_change(5) * 100
    df['btc_regime_bull'] = ((btc_slope > 0.5) & (df['btc_close'] > btc_sma50)).astype(int)
    df['btc_regime_bear'] = ((btc_slope < -0.5) & (df['btc_close'] < btc_sma50)).astype(int)

    # --- Relative volatility ---
    eth_vol = eth_ret.rolling(20).std()
    btc_vol = btc_ret.rolling(20).std()
    df['eth_btc_vol_ratio'] = eth_vol / (btc_vol + 1e-10)

    # --- BTC RSI ---
    df['btc_rsi_14'] = ta.momentum.RSIIndicator(df['btc_close'], window=14).rsi()
    df['btc_rsi_diff'] = df['1d_rsi_14'] - df['btc_rsi_14']  # ETH RSI vs BTC RSI

    # --- BTC leads ETH (lag features) ---
    df['btc_ret_lag1'] = btc_ret.shift(1)  # Yesterday's BTC return
    df['btc_ret_lag2'] = btc_ret.shift(2)
    df['btc_big_move_up'] = (btc_ret > 0.03).astype(int)  # BTC +3% day
    df['btc_big_move_down'] = (btc_ret < -0.03).astype(int)  # BTC -3% day

    # --- ETH beta to BTC (rolling) ---
    cov = eth_ret.rolling(30).cov(btc_ret)
    var = btc_ret.rolling(30).var()
    df['eth_btc_beta_30'] = cov / (var + 1e-10)

    # Cleanup temp columns
    df.drop(columns=['btc_close', 'btc_volume', 'btc_high', 'btc_low'], inplace=True, errors='ignore')

    n_btc_feats = len([c for c in df.columns if 'btc' in c.lower() or 'eth_btc' in c.lower()])
    logger.info(f"  Added {n_btc_feats} BTC influence features")

    return df


def create_labels(df):
    """Create regime-aware ATR-adaptive triple barrier labels"""
    atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
    median_atr = atr.rolling(50).median()

    sma_50 = df['close'].rolling(50).mean()
    sma_dist = (df['close'] / sma_50 - 1)
    rsi = ta.momentum.RSIIndicator(df['close'], window=14).rsi()

    labels = []
    n_suppressed = 0
    for i in range(len(df)):
        if i >= len(df) - 1:
            labels.append(-1)
            continue

        entry = df.iloc[i]['close']
        current_atr = atr.iloc[i]

        if pd.notna(current_atr) and current_atr > 0:
            tp_dist = current_atr * ATR_TP_MULT
            sl_dist = current_atr * ATR_SL_MULT
        else:
            tp_dist = entry * FIXED_TP_PCT
            sl_dist = entry * FIXED_SL_PCT

        tp = entry + tp_dist
        sl = entry - sl_dist

        med = median_atr.iloc[i] if pd.notna(median_atr.iloc[i]) and median_atr.iloc[i] > 0 else current_atr
        if pd.notna(med) and med > 0:
            vol_ratio = current_atr / med
            lookahead = int(BASE_LOOKAHEAD * max(0.5, min(2.0, vol_ratio * ATR_LOOKAHEAD_MULT + 0.3)))
        else:
            lookahead = BASE_LOOKAHEAD
        lookahead = max(5, min(20, lookahead))

        hit_tp = False
        hit_sl = False

        for j in range(i + 1, min(i + 1 + lookahead, len(df))):
            if df.iloc[j]['high'] >= tp:
                hit_tp = True
                break
            if df.iloc[j]['low'] <= sl:
                hit_sl = True
                break

        cur_sma_dist = sma_dist.iloc[i] if pd.notna(sma_dist.iloc[i]) else 0
        cur_rsi = rsi.iloc[i] if pd.notna(rsi.iloc[i]) else 50

        is_strong_bear = (cur_sma_dist < -0.10) or (cur_rsi < 30 and cur_sma_dist < -0.05)

        if hit_tp and is_strong_bear:
            labels.append(1)
            n_suppressed += 1
        elif hit_tp:
            labels.append(1)
        elif hit_sl:
            labels.append(0)
        else:
            labels.append(-1)

    df['label'] = labels

    df['is_strong_bear'] = (
        (sma_dist < -0.10) | ((rsi < 30) & (sma_dist < -0.05))
    ).astype(int)

    logger.info(f"  Labels: TP in bear market (downweighted): {n_suppressed}")
    logger.info(f"  Strong bear days: {df['is_strong_bear'].sum()}")
    return df


def build_features():
    """Main pipeline: merge timeframes, create features, save"""
    logger.info(f"\n{'='*70}")
    logger.info(f"ETH MULTI-TF FEATURE ENGINEERING (V3)")
    logger.info(f"{'='*70}\n")

    # Load 1d as base
    df_1d = pd.read_csv(DATA_DIR / f'{CRYPTO.lower()}_1d_data.csv')
    df_1d['date'] = pd.to_datetime(df_1d['date'])
    logger.info(f"Base 1d data: {len(df_1d)} candles")

    # Create 1d technical indicators
    df_1d = create_technical_indicators(df_1d, '1d_')

    # Load and merge other timeframes (4h, 1w only — no 1h noise)
    for tf in ['4h', '1w']:
        tf_file = DATA_DIR / f'{CRYPTO.lower()}_{tf}_data.csv'
        if not tf_file.exists():
            logger.warning(f"  {tf} data not found, skipping")
            continue

        df_tf = pd.read_csv(tf_file)
        df_tf['date'] = pd.to_datetime(df_tf['date'])
        df_tf = create_technical_indicators(df_tf, f'{tf}_')

        tf_cols = ['date'] + [c for c in df_tf.columns if c.startswith(f'{tf}_')]
        df_1d = pd.merge_asof(
            df_1d.sort_values('date'),
            df_tf[tf_cols].sort_values('date'),
            on='date',
            direction='backward'
        )
        logger.info(f"  Merged {tf}: {len(df_1d)} rows, +{len(tf_cols)-1} features")

    # Cross-TF features
    df_1d = create_cross_tf_features(df_1d)

    # Non-technical features
    df_1d = create_non_technical_features(df_1d)

    # Market regime features
    df_1d = create_market_regime_features(df_1d)

    # BTC influence features
    logger.info("  Adding BTC influence features...")
    df_1d = create_btc_influence_features(df_1d)

    # Labels
    logger.info("  Creating labels...")
    df_1d = create_labels(df_1d)

    # Identify feature columns
    exclude = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'date', 'label',
               'label_class', 'triple_barrier_label', 'returns']
    feature_cols = [c for c in df_1d.columns if c not in exclude and not c.startswith('Unnamed')]

    # Clean inf/nan
    for c in feature_cols:
        df_1d[c] = pd.to_numeric(df_1d[c], errors='coerce')
    df_1d[feature_cols] = df_1d[feature_cols].replace([np.inf, -np.inf], np.nan)

    # Save features list
    with open(BASE_DIR / 'required_features.json', 'w') as f:
        json.dump(feature_cols, f, indent=2)

    # Save merged data
    output_file = DATA_DIR / 'eth_features.csv'
    df_1d.to_csv(output_file, index=False)

    # Stats
    n_labeled = (df_1d['label'] != -1).sum()
    n_tp = (df_1d['label'] == 1).sum()
    n_sl = (df_1d['label'] == 0).sum()

    logger.info(f"\n{'='*70}")
    logger.info(f"FEATURE ENGINEERING COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"  Total rows: {len(df_1d)}")
    logger.info(f"  Total features: {len(feature_cols)}")
    logger.info(f"  Labeled: {n_labeled} (TP={n_tp}, SL={n_sl})")
    logger.info(f"  Saved to {output_file}")
    logger.info(f"  Features list saved to required_features.json")

    return df_1d


if __name__ == "__main__":
    build_features()
