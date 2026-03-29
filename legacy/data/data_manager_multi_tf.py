"""
Multi-Timeframe Data Manager for Crypto V10
Downloads and caches data for 4h, 1d, 1w timeframes
Optimized for multi-timeframe analysis
"""

import os
import json
import time
import requests
import pandas as pd
from datetime import datetime, timedelta

# Configuration
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
CACHE_VALIDITY_HOURS = 24  # Recharger si cache > 24h

# Cryptos configuration
CRYPTOS = {
    'BTC': 'BTCUSDT',
    'ETH': 'ETHUSDT',
    'SOL': 'SOLUSDT'
}

# Timeframes configuration - MAXIMIZE HISTORICAL DATA
TIMEFRAMES = {
    '4h': {
        'binance_interval': '4h',
        'limit': 10000,  # ~4.5 years (6 candles/day × 365 × 4.5)
        'description': 'Short-term (4 hours)'
    },
    '1d': {
        'binance_interval': '1d',
        'limit': 3000,  # ~8.2 years (back to 2017-2018)
        'description': 'Medium-term (1 day)'
    },
    '1w': {
        'binance_interval': '1w',
        'limit': 500,   # ~9.6 years (already maxed)
        'description': 'Long-term (1 week)'
    }
}


def ensure_cache_dir():
    """Create cache directory if doesn't exist"""
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
        print(f"[INFO] Cache directory created: {CACHE_DIR}")


def get_cache_path(crypto, timeframe):
    """Get cache file path: bitcoin_4h.json, ethereum_1d.json, etc."""
    crypto_lower = crypto.lower()
    return os.path.join(CACHE_DIR, f"{crypto_lower}_{timeframe}.json")


def is_cache_valid(cache_path, max_age_hours=CACHE_VALIDITY_HOURS):
    """Check if cache is still valid"""
    if not os.path.exists(cache_path):
        return False

    file_time = os.path.getmtime(cache_path)
    age_hours = (time.time() - file_time) / 3600

    return age_hours < max_age_hours


def download_binance_data(symbol, interval='1d', limit=2000):
    """
    Download data from Binance API

    Args:
        symbol: Trading pair (e.g., BTCUSDT)
        interval: Timeframe (4h, 1d, 1w)
        limit: Number of candles to fetch

    Returns:
        list: Raw Binance klines data
    """
    print(f"  [>>] Downloading from Binance: {symbol} ({interval}, {limit} candles)...")

    url = 'https://api.binance.com/api/v3/klines'
    all_data = []
    remaining = limit

    while remaining > 0:
        batch_size = min(1000, remaining)
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': batch_size
        }

        if all_data:
            # Continue from last timestamp
            params['endTime'] = all_data[0][0] - 1

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            batch_data = response.json()

            if not batch_data or len(batch_data) == 0:
                break

            # Add at beginning (reverse chronological)
            all_data = batch_data + all_data
            remaining -= len(batch_data)

            if len(batch_data) < batch_size:
                break

            time.sleep(0.2)  # Rate limiting

        except Exception as e:
            print(f"  [ERROR] Download failed: {e}")
            break

    print(f"  [OK] Downloaded {len(all_data)} candles")
    return all_data


def parse_binance_data(raw_data):
    """
    Parse Binance raw data into structured format

    Returns:
        dict: {
            'timestamps': [...],
            'dates': [...],
            'open': [...],
            'high': [...],
            'low': [...],
            'close': [...],
            'volume': [...]
        }
    """
    parsed = {
        'timestamps': [],
        'dates': [],
        'open': [],
        'high': [],
        'low': [],
        'close': [],
        'volume': []
    }

    for candle in raw_data:
        timestamp = candle[0]
        date = datetime.fromtimestamp(timestamp / 1000).strftime('%Y-%m-%d %H:%M:%S')

        parsed['timestamps'].append(timestamp)
        parsed['dates'].append(date)
        parsed['open'].append(float(candle[1]))
        parsed['high'].append(float(candle[2]))
        parsed['low'].append(float(candle[3]))
        parsed['close'].append(float(candle[4]))
        parsed['volume'].append(float(candle[5]))

    return parsed


def save_to_cache(data, cache_path):
    """Save data to cache file"""
    with open(cache_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  [SAVE] Cached to: {cache_path}")


def load_from_cache(cache_path):
    """Load data from cache file"""
    with open(cache_path, 'r') as f:
        data = json.load(f)
    print(f"  [CACHE] Loaded from: {cache_path}")
    return data


def fetch_crypto_timeframe(crypto, timeframe, force_download=False):
    """
    Fetch or load data for a specific crypto and timeframe

    Args:
        crypto: Crypto symbol (BTC, ETH, SOL)
        timeframe: Timeframe (4h, 1d, 1w)
        force_download: Force download even if cache valid

    Returns:
        dict: Parsed data
    """
    ensure_cache_dir()

    symbol = CRYPTOS[crypto]
    tf_config = TIMEFRAMES[timeframe]
    cache_path = get_cache_path(crypto, timeframe)

    print(f"\n[{crypto}] Fetching {timeframe} data...")

    # Check cache
    if not force_download and is_cache_valid(cache_path):
        return load_from_cache(cache_path)

    # Download from Binance
    raw_data = download_binance_data(
        symbol=symbol,
        interval=tf_config['binance_interval'],
        limit=tf_config['limit']
    )

    if not raw_data:
        print(f"  [ERROR] No data downloaded for {crypto} {timeframe}")
        return None

    # Parse and cache
    parsed_data = parse_binance_data(raw_data)
    save_to_cache(parsed_data, cache_path)

    return parsed_data


def fetch_all_data(cryptos=None, timeframes=None, force_download=False):
    """
    Fetch data for multiple cryptos and timeframes

    Args:
        cryptos: List of crypto symbols (default: all)
        timeframes: List of timeframes (default: all)
        force_download: Force download even if cache valid

    Returns:
        dict: {
            'BTC': {'4h': {...}, '1d': {...}, '1w': {...}},
            'ETH': {...},
            'SOL': {...}
        }
    """
    if cryptos is None:
        cryptos = list(CRYPTOS.keys())

    if timeframes is None:
        timeframes = list(TIMEFRAMES.keys())

    print("="*70)
    print("MULTI-TIMEFRAME DATA MANAGER V10")
    print("="*70)
    print(f"Cryptos: {', '.join(cryptos)}")
    print(f"Timeframes: {', '.join(timeframes)}")
    print(f"Force download: {force_download}")
    print("="*70)

    all_data = {}

    for crypto in cryptos:
        all_data[crypto] = {}

        for tf in timeframes:
            data = fetch_crypto_timeframe(crypto, tf, force_download)
            all_data[crypto][tf] = data

            if data:
                print(f"  [SUCCESS] {crypto} {tf}: {len(data['dates'])} candles "
                      f"({data['dates'][0]} to {data['dates'][-1]})")

    print("\n" + "="*70)
    print("DOWNLOAD SUMMARY")
    print("="*70)

    for crypto in cryptos:
        print(f"\n{crypto}:")
        for tf in timeframes:
            if all_data[crypto][tf]:
                count = len(all_data[crypto][tf]['dates'])
                first = all_data[crypto][tf]['dates'][0]
                last = all_data[crypto][tf]['dates'][-1]
                print(f"  {tf:3s}: {count:4d} candles ({first} to {last})")
            else:
                print(f"  {tf:3s}: FAILED")

    print("="*70)

    return all_data


def get_dataframe(crypto, timeframe):
    """
    Get data as pandas DataFrame

    Args:
        crypto: Crypto symbol (BTC, ETH, SOL)
        timeframe: Timeframe (4h, 1d, 1w)

    Returns:
        pd.DataFrame: Data with datetime index
    """
    cache_path = get_cache_path(crypto, timeframe)

    if not os.path.exists(cache_path):
        print(f"[ERROR] No cached data for {crypto} {timeframe}. Run fetch first.")
        return None

    data = load_from_cache(cache_path)

    df = pd.DataFrame({
        'open': data['open'],
        'high': data['high'],
        'low': data['low'],
        'close': data['close'],
        'volume': data['volume']
    })

    df['date'] = pd.to_datetime(data['dates'])
    df.set_index('date', inplace=True)

    return df


def validate_data_alignment(cryptos=['BTC', 'ETH', 'SOL'], timeframe='1d'):
    """
    Validate that data for multiple cryptos is temporally aligned
    Important for BTC influence features

    Args:
        cryptos: List of crypto symbols
        timeframe: Timeframe to check

    Returns:
        bool: True if aligned
    """
    print(f"\n[VALIDATION] Checking data alignment for {timeframe}...")

    dfs = {}
    for crypto in cryptos:
        df = get_dataframe(crypto, timeframe)
        if df is None:
            print(f"  [ERROR] Missing data for {crypto}")
            return False
        dfs[crypto] = df

    # Check date ranges
    print(f"\nDate ranges:")
    for crypto, df in dfs.items():
        print(f"  {crypto}: {df.index[0]} to {df.index[-1]} ({len(df)} rows)")

    # Check if dates match
    ref_dates = set(dfs[cryptos[0]].index)
    for crypto in cryptos[1:]:
        crypto_dates = set(dfs[crypto].index)

        only_in_ref = ref_dates - crypto_dates
        only_in_crypto = crypto_dates - ref_dates

        if only_in_ref or only_in_crypto:
            print(f"\n  [WARNING] Date mismatch between {cryptos[0]} and {crypto}")
            if only_in_ref:
                print(f"    Only in {cryptos[0]}: {len(only_in_ref)} dates")
            if only_in_crypto:
                print(f"    Only in {crypto}: {len(only_in_crypto)} dates")
        else:
            print(f"  [OK] {crypto} aligned with {cryptos[0]}")

    return True


if __name__ == "__main__":
    import sys

    # Parse command line arguments
    force_download = '--force' in sys.argv or '-f' in sys.argv

    # Fetch all data for BTC, ETH, SOL across 4h, 1d, 1w
    all_data = fetch_all_data(
        cryptos=['BTC', 'ETH', 'SOL'],
        timeframes=['4h', '1d', '1w'],
        force_download=force_download
    )

    # Validate alignment for 1d (most important for cross-crypto features)
    validate_data_alignment(cryptos=['BTC', 'ETH', 'SOL'], timeframe='1d')

    print("\n[DONE] Multi-timeframe data collection complete!")
    print(f"[INFO] Data cached in: {CACHE_DIR}")
    print(f"[INFO] Use get_dataframe(crypto, timeframe) to load as pandas DataFrame")
