"""
V12 Sentiment Features
=======================
Fetch and integrate:
1. Fear & Greed Index (API gratuite alternative.me)
2. Binance Funding Rates
3. Binance Open Interest

These capture market psychology that technical indicators miss.
"""

import sys
import numpy as np
import pandas as pd
import requests
import json
import time
from pathlib import Path
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).parent.parent.parent
CACHE_DIR = PROJECT_ROOT / 'data' / 'cache'


# ============================================================================
# FEAR & GREED INDEX
# ============================================================================

def fetch_fear_greed(limit=0) -> pd.DataFrame:
    """
    Fetch Fear & Greed Index from alternative.me API.
    Returns daily data: value (0-100), classification.
    0 = Extreme Fear, 100 = Extreme Greed
    """
    cache_file = CACHE_DIR / 'fear_greed.json'

    # Check cache (24h validity)
    if cache_file.exists():
        mtime = cache_file.stat().st_mtime
        if time.time() - mtime < 86400:
            with open(cache_file) as f:
                data = json.load(f)
            print(f"  [CACHE] Fear & Greed: {len(data)} days")
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df['date'] = pd.to_datetime(df['timestamp'], unit='s')
            df = df.set_index('date').sort_index()
            return df

    print(f"  [API] Fetching Fear & Greed Index...")
    try:
        url = "https://api.alternative.me/fng/?limit=0&format=json"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        raw = resp.json()['data']

        # Save cache
        with open(cache_file, 'w') as f:
            json.dump(raw, f)

        print(f"  Fetched {len(raw)} days")

        df = pd.DataFrame(raw)
        df['timestamp'] = df['timestamp'].astype(int)
        df['date'] = pd.to_datetime(df['timestamp'], unit='s')
        df['value'] = df['value'].astype(float)
        df = df.set_index('date').sort_index()
        return df

    except Exception as e:
        print(f"  Fear & Greed API error: {e}")
        return pd.DataFrame()


# ============================================================================
# BINANCE FUNDING RATES
# ============================================================================

def fetch_funding_rates(symbol: str = 'BTCUSDT', limit: int = 1000) -> pd.DataFrame:
    """
    Fetch historical funding rates from Binance Futures API.
    Funding rate = sentiment indicator (positive = longs pay shorts = bullish crowding).
    """
    cache_file = CACHE_DIR / f'funding_{symbol.lower()}.json'

    if cache_file.exists():
        mtime = cache_file.stat().st_mtime
        if time.time() - mtime < 86400:
            with open(cache_file) as f:
                data = json.load(f)
            print(f"  [CACHE] Funding {symbol}: {len(data)} entries")
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['fundingTime'], unit='ms')
            df['fundingRate'] = df['fundingRate'].astype(float)
            df = df.set_index('date').sort_index()
            return df

    print(f"  [API] Fetching Funding Rates {symbol}...")
    all_data = []
    end_time = None

    try:
        for _ in range(10):  # Max 10 pages
            url = "https://fapi.binance.com/fapi/v1/fundingRate"
            params = {'symbol': symbol, 'limit': 1000}
            if end_time:
                params['endTime'] = end_time

            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            if not data:
                break

            all_data.extend(data)
            end_time = data[0]['fundingTime'] - 1

            if len(data) < 1000:
                break

            time.sleep(0.2)

        # Save cache
        with open(cache_file, 'w') as f:
            json.dump(all_data, f)

        print(f"  Fetched {len(all_data)} funding entries")

        df = pd.DataFrame(all_data)
        df['date'] = pd.to_datetime(df['fundingTime'], unit='ms')
        df['fundingRate'] = df['fundingRate'].astype(float)
        df = df.set_index('date').sort_index()
        return df

    except Exception as e:
        print(f"  Funding API error: {e}")
        return pd.DataFrame()


# ============================================================================
# BINANCE OPEN INTEREST
# ============================================================================

def fetch_open_interest(symbol: str = 'BTCUSDT', period: str = '1d', limit: int = 500) -> pd.DataFrame:
    """
    Fetch historical open interest from Binance Futures.
    OI increasing + price increasing = strong trend.
    OI increasing + price decreasing = potential reversal.
    """
    cache_file = CACHE_DIR / f'oi_{symbol.lower()}_{period}.json'

    if cache_file.exists():
        mtime = cache_file.stat().st_mtime
        if time.time() - mtime < 86400:
            with open(cache_file) as f:
                data = json.load(f)
            print(f"  [CACHE] OI {symbol}: {len(data)} entries")
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['sumOpenInterest'] = df['sumOpenInterest'].astype(float)
            df['sumOpenInterestValue'] = df['sumOpenInterestValue'].astype(float)
            df = df.set_index('date').sort_index()
            return df

    print(f"  [API] Fetching Open Interest {symbol}...")
    try:
        url = "https://fapi.binance.com/futures/data/openInterestHist"
        params = {'symbol': symbol, 'period': period, 'limit': limit}

        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        with open(cache_file, 'w') as f:
            json.dump(data, f)

        print(f"  Fetched {len(data)} OI entries")

        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['sumOpenInterest'] = df['sumOpenInterest'].astype(float)
        df['sumOpenInterestValue'] = df['sumOpenInterestValue'].astype(float)
        df = df.set_index('date').sort_index()
        return df

    except Exception as e:
        print(f"  OI API error: {e}")
        return pd.DataFrame()


# ============================================================================
# BUILD SENTIMENT FEATURES
# ============================================================================

def build_sentiment_features(crypto: str, df_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Build sentiment features aligned to price data index.

    Returns DataFrame with columns:
    - fg_value: Fear & Greed (0-100)
    - fg_ma7: 7-day MA of F&G
    - fg_extreme_fear: 1 if F&G < 25
    - fg_extreme_greed: 1 if F&G > 75
    - fg_change_7d: F&G change over 7 days
    - funding_rate: latest funding rate (8h, resampled to daily)
    - funding_ma7: 7-day MA of funding
    - funding_positive: 1 if funding > 0 (bullish crowding)
    - oi_change_pct: OI % change (daily)
    - oi_ma7_change: 7-day MA of OI change
    """
    symbol_map = {'btc': 'BTCUSDT', 'eth': 'ETHUSDT', 'sol': 'SOLUSDT'}
    symbol = symbol_map.get(crypto, 'BTCUSDT')

    result = pd.DataFrame(index=df_prices.index)

    # 1. Fear & Greed
    print(f"\n  Sentiment features for {crypto.upper()}:")
    fg = fetch_fear_greed()
    if len(fg) > 0:
        fg_daily = fg[['value']].resample('1D').last().ffill()
        fg_aligned = fg_daily.reindex(df_prices.index, method='ffill')

        result['fg_value'] = fg_aligned['value']
        result['fg_ma7'] = result['fg_value'].rolling(7).mean()
        result['fg_extreme_fear'] = (result['fg_value'] < 25).astype(int)
        result['fg_extreme_greed'] = (result['fg_value'] > 75).astype(int)
        result['fg_change_7d'] = result['fg_value'].diff(7)
        print(f"    Fear & Greed: {result['fg_value'].notna().sum()} days mapped")
    else:
        for col in ['fg_value', 'fg_ma7', 'fg_extreme_fear', 'fg_extreme_greed', 'fg_change_7d']:
            result[col] = np.nan

    # 2. Funding Rates
    fr = fetch_funding_rates(symbol)
    if len(fr) > 0:
        fr_daily = fr[['fundingRate']].resample('1D').mean()
        fr_aligned = fr_daily.reindex(df_prices.index, method='ffill')

        result['funding_rate'] = fr_aligned['fundingRate']
        result['funding_ma7'] = result['funding_rate'].rolling(7).mean()
        result['funding_positive'] = (result['funding_rate'] > 0).astype(int)
        print(f"    Funding: {result['funding_rate'].notna().sum()} days mapped")
    else:
        for col in ['funding_rate', 'funding_ma7', 'funding_positive']:
            result[col] = np.nan

    # 3. Open Interest
    oi = fetch_open_interest(symbol, period='1d')
    if len(oi) > 0:
        oi_daily = oi[['sumOpenInterestValue']].resample('1D').last().ffill()
        oi_aligned = oi_daily.reindex(df_prices.index, method='ffill')

        result['oi_value'] = oi_aligned['sumOpenInterestValue']
        result['oi_change_pct'] = result['oi_value'].pct_change() * 100
        result['oi_ma7_change'] = result['oi_change_pct'].rolling(7).mean()
        print(f"    Open Interest: {result['oi_value'].notna().sum()} days mapped")
    else:
        for col in ['oi_value', 'oi_change_pct', 'oi_ma7_change']:
            result[col] = np.nan

    # Fill NaN
    result = result.ffill().fillna(0)

    # Stats
    n_features = len(result.columns)
    n_valid = result.notna().all(axis=1).sum()
    print(f"    Total: {n_features} sentiment features, {n_valid} complete rows")

    return result


SENTIMENT_COLS = [
    'fg_value', 'fg_ma7', 'fg_extreme_fear', 'fg_extreme_greed', 'fg_change_7d',
    'funding_rate', 'funding_ma7', 'funding_positive',
    'oi_value', 'oi_change_pct', 'oi_ma7_change'
]
