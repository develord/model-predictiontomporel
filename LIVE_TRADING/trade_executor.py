"""
Trade Executor - Binance Futures Testnet
=========================================
Places market orders with TP/SL on Binance Futures Testnet.
TP/SL are placed as limit/stop orders on the exchange.
"""

import ccxt
import logging
import time

from config import BINANCE_TESTNET_KEY, BINANCE_TESTNET_SECRET, TRADING, COINS

logger = logging.getLogger(__name__)


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

            # Test connection - fetch balance directly
            balances = self.exchange.fapiPrivateV2GetBalance()
            usdt_bal = next((b for b in balances if b['asset'] == 'USDT'), None)
            usdt = float(usdt_bal['availableBalance']) if usdt_bal else 0
            logger.info(f"Connected to Binance Demo Trading | USDT Balance: {usdt}")
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
                price_precision = int(sym_info.get('pricePrecision', 2))
                min_qty = float(next((f['minQty'] for f in sym_info.get('filters', []) if f['filterType'] == 'LOT_SIZE'), 0.001))
            else:
                qty_precision = 3
                price_precision = 2
                min_qty = 0.001
            precision = qty_precision

            quantity = round(position_value / price, precision)
            if quantity < min_qty:
                logger.warning(f"{coin}: Quantity {quantity} below minimum {min_qty}")
                return None

            # TP/SL prices
            tp_price = round(price * (1 + tp_pct), price_precision)
            sl_price = round(price * (1 - sl_pct), price_precision)

            logger.info(f"{coin}: Opening LONG | Price: {price} | Qty: {quantity} | TP: {tp_price} | SL: {sl_price}")

            # 1. Market buy order
            entry_order = self.exchange.fapiPrivatePostOrder({
                'symbol': symbol,
                'side': 'BUY',
                'type': 'MARKET',
                'quantity': quantity,
            })
            entry_price = float(entry_order.get('avgPrice', price))
            logger.info(f"{coin}: Entry filled @ {entry_price}")

            # 2. Take Profit order
            tp_order = None
            try:
                tp_order = self.exchange.fapiPrivatePostOrder({
                    'symbol': symbol,
                    'side': 'SELL',
                    'type': 'TAKE_PROFIT_MARKET',
                    'stopPrice': tp_price,
                    'quantity': quantity,
                    'reduceOnly': 'true',
                })
                logger.info(f"{coin}: TP order placed @ {tp_price}")
            except Exception as e:
                logger.warning(f"{coin}: TP order failed ({e})")

            # 3. Stop Loss order
            sl_order = None
            try:
                sl_order = self.exchange.fapiPrivatePostOrder({
                    'symbol': symbol,
                    'side': 'SELL',
                    'type': 'STOP_MARKET',
                    'stopPrice': sl_price,
                    'quantity': quantity,
                    'reduceOnly': 'true',
                })
                logger.info(f"{coin}: SL order placed @ {sl_price}")
            except Exception as e:
                logger.warning(f"{coin}: SL order failed ({e})")

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
                qty_precision = int(sym_info.get('quantityPrecision', 3))
                price_precision = int(sym_info.get('pricePrecision', 2))
                min_qty = float(next((f['minQty'] for f in sym_info.get('filters', []) if f['filterType'] == 'LOT_SIZE'), 0.001))
            else:
                qty_precision, price_precision, min_qty = 3, 2, 0.001

            quantity = round(position_value / price, qty_precision)
            if quantity < min_qty:
                logger.warning(f"{coin}: Quantity below minimum")
                return None

            # SHORT: TP when price drops, SL when price rises
            tp_price = round(price * (1 - tp_pct), price_precision)
            sl_price = round(price * (1 + sl_pct), price_precision)

            logger.info(f"{coin}: Opening SHORT | Price: {price} | Qty: {quantity} | TP: {tp_price} | SL: {sl_price}")

            # Market SELL to open short
            entry_order = self.exchange.fapiPrivatePostOrder({
                'symbol': symbol, 'side': 'SELL', 'type': 'MARKET', 'quantity': quantity,
            })
            entry_price = float(entry_order.get('avgPrice', price))
            logger.info(f"{coin}: SHORT entry filled @ {entry_price}")

            # TP order (buy back at lower price)
            tp_order = None
            try:
                tp_order = self.exchange.fapiPrivatePostOrder({
                    'symbol': symbol, 'side': 'BUY', 'type': 'TAKE_PROFIT_MARKET',
                    'stopPrice': tp_price, 'quantity': quantity, 'reduceOnly': 'true',
                })
                logger.info(f"{coin}: SHORT TP @ {tp_price}")
            except Exception as e:
                logger.warning(f"{coin}: SHORT TP failed ({e})")

            # SL order (buy back at higher price)
            sl_order = None
            try:
                sl_order = self.exchange.fapiPrivatePostOrder({
                    'symbol': symbol, 'side': 'BUY', 'type': 'STOP_MARKET',
                    'stopPrice': sl_price, 'quantity': quantity, 'reduceOnly': 'true',
                })
                logger.info(f"{coin}: SHORT SL @ {sl_price}")
            except Exception as e:
                logger.warning(f"{coin}: SHORT SL failed ({e})")

            return {
                'coin': coin, 'pair': pair, 'side': 'SHORT',
                'entry_price': entry_price, 'quantity': quantity,
                'tp_price': tp_price, 'sl_price': sl_price,
                'tp_pct': tp_pct, 'sl_pct': sl_pct,
            }

        except Exception as e:
            logger.error(f"{coin}: SHORT execution failed: {e}")
            return None

    def close_position(self, coin, quantity):
        """Close position with market sell"""
        try:
            pair = COINS[coin]['pair']
            order = self.exchange.create_order(
                symbol=pair,
                type='market',
                side='sell',
                amount=quantity,
                params={'reduceOnly': True}
            )
            logger.info(f"{coin}: Position closed @ {order.get('average', 'N/A')}")
            return order
        except Exception as e:
            logger.error(f"{coin}: Close position failed: {e}")
            return None

    def cancel_orders(self, coin):
        """Cancel all open orders for a coin"""
        try:
            pair = COINS[coin]['pair']
            self.exchange.cancel_all_orders(pair)
            logger.info(f"{coin}: All open orders cancelled")
        except Exception as e:
            logger.warning(f"{coin}: Cancel orders failed: {e}")

    def get_open_positions(self):
        """Get all open positions"""
        try:
            positions = self.exchange.fetch_positions()
            open_pos = {}
            for pos in positions:
                if float(pos.get('contracts', 0)) > 0:
                    symbol = pos.get('symbol', '')
                    open_pos[symbol] = {
                        'side': pos.get('side'),
                        'contracts': float(pos.get('contracts', 0)),
                        'entry_price': float(pos.get('entryPrice', 0)),
                        'unrealized_pnl': float(pos.get('unrealizedPnl', 0)),
                    }
            return open_pos
        except Exception as e:
            logger.error(f"Fetch positions error: {e}")
            return {}
