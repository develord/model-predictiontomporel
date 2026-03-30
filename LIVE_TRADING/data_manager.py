"""
Data Manager - WebSocket streams + REST bootstrap + rolling buffers
====================================================================
"""

import asyncio
import json
import logging
import time
from datetime import datetime

import ccxt
import pandas as pd
import websockets

from config import COINS, BUFFER_SIZES, WS_RECONNECT_DELAY, WS_MAX_RECONNECT_DELAY

logger = logging.getLogger(__name__)


class DataManager:
    def __init__(self):
        self.buffers = {}  # {coin: {tf: DataFrame}}
        self.exchange = ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'spot'}})
        self.on_daily_close = None     # callback(coin)
        self.on_15m_close = None       # callback(coin, candle)
        self._ws = None
        self._running = False

    def bootstrap(self):
        """Fetch historical data to fill all buffers"""
        logger.info("Bootstrapping historical data...")

        for coin, cfg in COINS.items():
            self.buffers[coin] = {}
            pair = cfg['pair']

            for tf in cfg['timeframes'] + ['15m']:
                limit = BUFFER_SIZES.get(tf, 300)
                try:
                    ohlcv = self.exchange.fetch_ohlcv(pair, tf, limit=limit)
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
                    self.buffers[coin][tf] = df
                    logger.info(f"  {coin} {tf}: {len(df)} candles")
                except Exception as e:
                    logger.error(f"  {coin} {tf}: Bootstrap failed: {e}")
                    self.buffers[coin][tf] = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'date'])

        logger.info("Bootstrap complete")

    def _build_ws_url(self):
        """Build combined WebSocket stream URL"""
        streams = []
        for coin, cfg in COINS.items():
            symbol = cfg['pair'].replace('/', '').lower()
            for tf in cfg['timeframes'] + ['15m']:
                streams.append(f"{symbol}@kline_{tf}")

        stream_str = '/'.join(streams)
        return f"wss://stream.binance.com:9443/stream?streams={stream_str}"

    def _parse_ws_message(self, data):
        """Parse WebSocket kline message"""
        if 'data' not in data:
            return None

        msg = data['data']
        if msg.get('e') != 'kline':
            return None

        k = msg['k']
        symbol = msg['s'].upper()  # e.g. BTCUSDT
        tf = k['i']                # e.g. 1d, 4h, 15m
        is_closed = k['x']         # True if candle is final

        # Map symbol to coin
        coin = None
        for c, cfg in COINS.items():
            if cfg['pair'].replace('/', '') == symbol:
                coin = c
                break

        if not coin:
            return None

        candle = {
            'timestamp': k['t'],
            'open': float(k['o']),
            'high': float(k['h']),
            'low': float(k['l']),
            'close': float(k['c']),
            'volume': float(k['v']),
            'date': pd.Timestamp(k['t'], unit='ms'),
        }

        return coin, tf, is_closed, candle

    def _update_buffer(self, coin, tf, candle):
        """Append candle to buffer and trim"""
        if coin not in self.buffers or tf not in self.buffers[coin]:
            return

        new_row = pd.DataFrame([candle])
        buf = self.buffers[coin][tf]

        # Check if this candle already exists (update last row)
        if len(buf) > 0 and buf.iloc[-1]['timestamp'] == candle['timestamp']:
            self.buffers[coin][tf].iloc[-1] = new_row.iloc[0]
        else:
            max_size = BUFFER_SIZES.get(tf, 300)
            self.buffers[coin][tf] = pd.concat([buf, new_row], ignore_index=True).tail(max_size)

    async def run_websocket(self):
        """Main WebSocket loop with reconnection"""
        self._running = True
        delay = WS_RECONNECT_DELAY

        while self._running:
            try:
                url = self._build_ws_url()
                logger.info(f"Connecting to Binance WebSocket ({len(COINS)} coins)...")

                async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                    self._ws = ws
                    delay = WS_RECONNECT_DELAY
                    logger.info("WebSocket connected")

                    async for message in ws:
                        try:
                            data = json.loads(message)
                            result = self._parse_ws_message(data)
                            if not result:
                                continue

                            coin, tf, is_closed, candle = result

                            # Always update current candle
                            self._update_buffer(coin, tf, candle)

                            # On candle close
                            if is_closed:
                                if tf == '1d' and self.on_daily_close:
                                    logger.info(f"{coin} 1D candle closed @ {candle['close']}")
                                    await self.on_daily_close(coin)

                                if tf == '15m' and self.on_15m_close:
                                    await self.on_15m_close(coin, candle)

                        except json.JSONDecodeError:
                            continue
                        except Exception as e:
                            logger.error(f"WS message error: {e}")

            except (websockets.exceptions.ConnectionClosed, ConnectionError, OSError) as e:
                logger.warning(f"WebSocket disconnected: {e}")

            except Exception as e:
                logger.error(f"WebSocket error: {e}")

            if self._running:
                logger.info(f"Reconnecting in {delay}s...")
                await asyncio.sleep(delay)
                delay = min(delay * 2, WS_MAX_RECONNECT_DELAY)

    def stop(self):
        self._running = False
        if self._ws:
            asyncio.ensure_future(self._ws.close())
