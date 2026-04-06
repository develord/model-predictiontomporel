"""
${coin} Data Download Script
========================
Downloads multi-timeframe ETH data from Binance (1h, 4h, 1d, 1w)

Usage:
    python 01_download_data.py
"""

import ccxt
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data' / 'cache'
DATA_DIR.mkdir(parents=True, exist_ok=True)

CRYPTO = 'NEAR'
TIMEFRAMES = ['15m', '1h', '4h', '1d', '1w']
START_DATE = '2020-10-01'


def download_data(crypto, timeframe, since):
    """Download OHLCV data from Binance"""
    logger.info(f"Downloading {crypto} {timeframe} data since {since}...")

    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'}
    })

    symbol = f'{crypto}/USDT'
    since_ms = int(pd.Timestamp(since).timestamp() * 1000)

    all_ohlcv = []
    while True:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since_ms, limit=1000)

        if not ohlcv:
            break

        all_ohlcv.extend(ohlcv)
        since_ms = ohlcv[-1][0] + 1

        if len(ohlcv) < 1000:
            break

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')

    output_file = DATA_DIR / f'{crypto.lower()}_{timeframe}_data.csv'
    df.to_csv(output_file, index=False)

    logger.info(f"  ✓ Saved {len(df)} candles to {output_file}")
    return df


if __name__ == "__main__":
    logger.info(f"\n{'='*70}")
    logger.info(f"DOWNLOADING {CRYPTO} DATA")
    logger.info(f"{'='*70}\n")

    for tf in TIMEFRAMES:
        try:
            download_data(CRYPTO, tf, START_DATE)
        except Exception as e:
            logger.error(f"Error downloading {tf}: {e}")

    logger.info(f"\n✓ All {CRYPTO} data downloaded successfully!")
