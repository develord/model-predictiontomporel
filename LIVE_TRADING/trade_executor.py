"""
Trade Executor - Binance Futures Testnet
=========================================
Places market orders with TP/SL on Binance Futures Testnet.
TP/SL are placed as limit/stop orders on the exchange.
"""

import ccxt
import logging
import time
import hmac
import hashlib
import requests
from urllib.parse import urlencode

from config import BINANCE_TESTNET_KEY, BINANCE_TESTNET_SECRET, TRADING, COINS

DEMO_BASE_URL = 'https://demo-fapi.binance.com'

logger = logging.getLogger(__name__)


def round_to_tick(value, tick_size):
    """Round price to nearest tick size"""
    tick = float(tick_size)
    if tick <= 0:
        return value
    return round(round(value / tick) * tick, 10)


class TradeExecutor:
    def __init__(self):
        self.exchange = None
        self.connected = False

    def connect(self):
        """Connect to Binance Futures Demo Trading (demo-fapi.binance.com)"""
        try:
            self.exchange = ccxt.binance({
                'apiKey': BINANCE_TESTNET_KEY,
                'secret': BINANCE_TESTNET_SECRET,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future',
                    'adjustForTimeDifference': True,
                    'fetchCurrencies': False,
                },
            })

            # Redirect ALL API URLs to demo-fapi
            demo = 'https://demo-fapi.binance.com'
            for key in list(self.exchange.urls['api'].keys()):
                url = self.exchange.urls['api'][key]
                if isinstance(url, str):
                    if 'fapi.binance.com' in url:
                        self.exchange.urls['api'][key] = url.replace('https://fapi.binance.com', demo)
                    elif 'api.binance.com' in url:
                        self.exchange.urls['api'][key] = url.replace('https://api.binance.com', demo)

            # Set leverage to 1x for all coins
            for coin_name, coin_cfg in COINS.items():
                try:
                    sym = coin_cfg['pair'].replace('/', '')
                    self.exchange.fapiPrivatePostLeverage({'symbol': sym, 'leverage': TRADING['leverage']})
                except:
                    pass

            # Test connection - fetch balance directly
            balances = self.exchange.fapiPrivateV2GetBalance()
            usdt_bal = next((b for b in balances if b['asset'] == 'USDT'), None)
            usdt = float(usdt_bal['availableBalance']) if usdt_bal else 0
            logger.info(f"Connected to Binance Demo Trading | USDT Balance: {usdt} | Leverage: {TRADING['leverage']}x")
            self.connected = True
            return True

        except Exception as e:
            logger.error(f"Failed to connect to Binance Demo: {e}")
            logger.error("Get demo API keys from: https://www.binance.com/en/demo-trading")
            self.connected = False
            return False

    def get_balance(self):
        """Get USDT balance from demo"""
        try:
            balances = self.exchange.fapiPrivateV2GetBalance()
            usdt = next((b for b in balances if b['asset'] == 'USDT'), None)
            return float(usdt['availableBalance']) if usdt else 0
        except Exception as e:
            logger.error(f"Balance fetch error: {e}")
            return 0

    def get_price(self, coin):
        """Get current price from demo"""
        try:
            symbol = COINS[coin]['pair'].replace('/', '')
            ticker = self.exchange.fapiPublicGetTickerPrice({'symbol': symbol})
            return float(ticker['price'])
        except Exception as e:
            logger.error(f"{coin} price fetch error: {e}")
            return None

    def open_long(self, coin, tp_pct, sl_pct):
        """
        Open LONG position with TP and SL orders on exchange.
        Returns order info dict or None.
        """
        if not self.connected:
            logger.error("Not connected to exchange")
            return None

        pair = COINS[coin]['pair']
        symbol = pair.replace('/', '')

        try:
            # Get current price
            price = self.get_price(coin)
            if not price:
                return None

            # Calculate position size
            balance = self.get_balance()
            position_value = balance * TRADING['position_size_pct']

            # Get precision from exchange info (futures endpoint)
            exchange_info = self.exchange.fapiPublicGetExchangeInfo()
            sym_info = next((s for s in exchange_info['symbols'] if s['symbol'] == symbol), None)

            if sym_info:
                qty_precision = int(sym_info.get('quantityPrecision', 3))
                tick_size = next((f['tickSize'] for f in sym_info.get('filters', []) if f['filterType'] == 'PRICE_FILTER'), '0.01')
                step_size = next((f['stepSize'] for f in sym_info.get('filters', []) if f['filterType'] == 'LOT_SIZE'), '0.001')
                min_qty = float(next((f['minQty'] for f in sym_info.get('filters', []) if f['filterType'] == 'LOT_SIZE'), 0.001))
            else:
                qty_precision, tick_size, step_size, min_qty = 3, '0.01', '0.001', 0.001

            quantity = round_to_tick(position_value / price, step_size)
            if quantity < min_qty:
                logger.warning(f"{coin}: Quantity {quantity} below minimum {min_qty}")
                return None

            # TP/SL prices aligned to tick size
            tp_price = round_to_tick(price * (1 + tp_pct), tick_size)
            sl_price = round_to_tick(price * (1 - sl_pct), tick_size)

            logger.info(f"{coin}: Opening LONG | Price: {price} | Qty: {quantity} | TP: {tp_price} | SL: {sl_price}")

            # 1. Market buy order
            entry_order = self.exchange.fapiPrivatePostOrder({
                'symbol': symbol,
                'side': 'BUY',
                'type': 'MARKET',
                'quantity': quantity,
            })
            avg = float(entry_order.get('avgPrice', 0))
            entry_price = avg if avg > 0 else price  # Demo returns 0, use market price
            logger.info(f"{coin}: Entry filled @ {entry_price}")

            # 2. Take Profit (LIMIT sell at TP price)
            tp_order = None
            try:
                tp_order = self.exchange.fapiPrivatePostOrder({
                    'symbol': symbol, 'side': 'SELL', 'type': 'LIMIT',
                    'price': tp_price, 'quantity': quantity,
                    'timeInForce': 'GTC', 'reduceOnly': 'true',
                })
                logger.info(f"{coin}: TP LIMIT @ {tp_price}")
            except Exception as e:
                logger.warning(f"{coin}: TP failed ({e})")

            # 3. Stop Loss — try regular STOP_MARKET first, then algo fallback
            sl_order = None
            try:
                sl_order = self.exchange.fapiPrivatePostOrder({
                    'symbol': symbol, 'side': 'SELL', 'type': 'STOP_MARKET',
                    'stopPrice': sl_price, 'quantity': quantity,
                    'reduceOnly': 'true',
                })
                logger.info(f"{coin}: SL STOP_MARKET @ {sl_price}")
            except Exception as e1:
                logger.warning(f"{coin}: SL regular failed ({e1}), trying algo...")
                try:
                    sl_order = self._place_algo_order(symbol, 'SELL', 'STOP_MARKET', sl_price, quantity)
                    logger.info(f"{coin}: SL STOP_MARKET @ {sl_price} (Algo API)")
                except Exception as e2:
                    logger.error(f"{coin}: SL FAILED both methods ({e2}), monitored by system only!")

            return {
                'coin': coin,
                'pair': pair,
                'side': 'LONG',
                'entry_price': entry_price,
                'quantity': quantity,
                'tp_price': tp_price,
                'sl_price': sl_price,
                'tp_pct': tp_pct,
                'sl_pct': sl_pct,
                'entry_order_id': entry_order.get('id'),
                'tp_order_id': tp_order.get('id') if tp_order else None,
                'sl_order_id': sl_order.get('id') if sl_order else None,
            }

        except Exception as e:
            logger.error(f"{coin}: Order execution failed: {e}")
            return None

    def open_short(self, coin, tp_pct, sl_pct):
        """Open SHORT position with TP and SL orders. TP = price drops, SL = price rises."""
        if not self.connected:
            return None

        pair = COINS[coin]['pair']
        symbol = pair.replace('/', '')

        try:
            price = self.get_price(coin)
            if not price:
                return None

            balance = self.get_balance()
            position_value = balance * TRADING['position_size_pct']

            exchange_info = self.exchange.fapiPublicGetExchangeInfo()
            sym_info = next((s for s in exchange_info['symbols'] if s['symbol'] == symbol), None)

            if sym_info:
                tick_size = next((f['tickSize'] for f in sym_info.get('filters', []) if f['filterType'] == 'PRICE_FILTER'), '0.01')
                step_size = next((f['stepSize'] for f in sym_info.get('filters', []) if f['filterType'] == 'LOT_SIZE'), '0.001')
                min_qty = float(next((f['minQty'] for f in sym_info.get('filters', []) if f['filterType'] == 'LOT_SIZE'), 0.001))
            else:
                tick_size, step_size, min_qty = '0.01', '0.001', 0.001

            quantity = round_to_tick(position_value / price, step_size)
            if quantity < min_qty:
                logger.warning(f"{coin}: Quantity below minimum")
                return None

            # SHORT: TP when price drops, SL when price rises (aligned to tick)
            tp_price = round_to_tick(price * (1 - tp_pct), tick_size)
            sl_price = round_to_tick(price * (1 + sl_pct), tick_size)

            logger.info(f"{coin}: Opening SHORT | Price: {price} | Qty: {quantity} | TP: {tp_price} | SL: {sl_price}")

            # Market SELL to open short
            entry_order = self.exchange.fapiPrivatePostOrder({
                'symbol': symbol, 'side': 'SELL', 'type': 'MARKET', 'quantity': quantity,
            })
            avg = float(entry_order.get('avgPrice', 0))
            entry_price = avg if avg > 0 else price  # Demo returns 0, use market price
            logger.info(f"{coin}: SHORT entry filled @ {entry_price}")

            # TP order (LIMIT buy back at lower price)
            tp_order = None
            try:
                tp_order = self.exchange.fapiPrivatePostOrder({
                    'symbol': symbol, 'side': 'BUY', 'type': 'LIMIT',
                    'price': tp_price, 'quantity': quantity,
                    'timeInForce': 'GTC', 'reduceOnly': 'true',
                })
                logger.info(f"{coin}: SHORT TP LIMIT @ {tp_price}")
            except Exception as e:
                logger.warning(f"{coin}: SHORT TP failed ({e})")

            # SL — try regular STOP_MARKET first, then algo fallback
            sl_order = None
            try:
                sl_order = self.exchange.fapiPrivatePostOrder({
                    'symbol': symbol, 'side': 'BUY', 'type': 'STOP_MARKET',
                    'stopPrice': sl_price, 'quantity': quantity,
                    'reduceOnly': 'true',
                })
                logger.info(f"{coin}: SHORT SL STOP_MARKET @ {sl_price}")
            except Exception as e1:
                logger.warning(f"{coin}: SHORT SL regular failed ({e1}), trying algo...")
                try:
                    sl_order = self._place_algo_order(symbol, 'BUY', 'STOP_MARKET', sl_price, quantity)
                    logger.info(f"{coin}: SHORT SL STOP_MARKET @ {sl_price} (Algo API)")
                except Exception as e2:
                    logger.error(f"{coin}: SHORT SL FAILED both methods ({e2}), monitored by system only!")

            return {
                'coin': coin, 'pair': pair, 'side': 'SHORT',
                'entry_price': entry_price, 'quantity': quantity,
                'tp_price': tp_price, 'sl_price': sl_price,
                'tp_pct': tp_pct, 'sl_pct': sl_pct,
            }

        except Exception as e:
            logger.error(f"{coin}: SHORT execution failed: {e}")
            return None

    def _place_algo_order(self, symbol, side, order_type, trigger_price, quantity):
        """Place conditional order via Binance Algo Order API with reduceOnly"""
        params = {
            'symbol': symbol,
            'side': side,
            'type': order_type,
            'algoType': 'CONDITIONAL',
            'triggerPrice': str(trigger_price),
            'quantity': str(quantity),
            'reduceOnly': 'true',
            'timestamp': int(time.time() * 1000),
            'recvWindow': 10000,
        }
        query = urlencode(params)
        signature = hmac.new(BINANCE_TESTNET_SECRET.encode(), query.encode(), hashlib.sha256).hexdigest()
        params['signature'] = signature
        headers = {'X-MBX-APIKEY': BINANCE_TESTNET_KEY}

        r = requests.post(f'{DEMO_BASE_URL}/fapi/v1/algoOrder', params=params, headers=headers)
        if r.status_code == 200:
            return r.json()
        else:
            raise Exception(f"Algo order failed: {r.text[:200]}")

    def close_position(self, coin, quantity, direction='LONG'):
        """Close position - SELL to close LONG, BUY to close SHORT"""
        try:
            symbol = COINS[coin]['pair'].replace('/', '')
            side = 'SELL' if direction == 'LONG' else 'BUY'
            order = self.exchange.fapiPrivatePostOrder({
                'symbol': symbol, 'side': side, 'type': 'MARKET',
                'quantity': quantity, 'reduceOnly': 'true',
            })
            avg_price = float(order.get('avgPrice', 0) or 0)
            # Binance Demo bug: avgPrice may return 0 even on success — retry from trade history
            if avg_price == 0:
                time.sleep(1)
                try:
                    trades = self.exchange.fapiPrivateGetUserTrades({'symbol': symbol, 'limit': 1})
                    if trades:
                        avg_price = float(trades[-1].get('price', 0))
                except Exception:
                    pass
            logger.info(f"{coin}: Position closed ({side}) @ {avg_price if avg_price else 'N/A'}")
            order['_fill_price'] = avg_price
            return order
        except Exception as e:
            logger.error(f"{coin}: Close position failed: {e}")
            return None

    def cancel_orders(self, coin):
        """Cancel all open orders for a coin"""
        try:
            symbol = COINS[coin]['pair'].replace('/', '')
            self.exchange.fapiPrivateDeleteAllOpenOrders({'symbol': symbol})
            logger.info(f"{coin}: Orders cancelled")
        except Exception as e:
            logger.warning(f"{coin}: Cancel orders: {e}")

    def get_open_positions(self):
        """Get all open positions from demo API"""
        try:
            positions = self.exchange.fapiPrivateV2GetPositionRisk()
            open_pos = {}
            for pos in positions:
                amt = float(pos.get('positionAmt', 0))
                if amt != 0:
                    symbol = pos.get('symbol', '')
                    open_pos[symbol] = {
                        'side': 'SHORT' if amt < 0 else 'LONG',
                        'contracts': abs(amt),
                        'entry_price': float(pos.get('entryPrice', 0)),
                        'unrealized_pnl': float(pos.get('unRealizedProfit', 0)),
                    }
            return open_pos
        except Exception as e:
            logger.error(f"Fetch positions error: {e}")
            return None  # None = error, {} = no positions
