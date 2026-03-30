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
        """Connect to Binance Futures Testnet"""
        try:
            self.exchange = ccxt.binance({
                'apiKey': BINANCE_TESTNET_KEY,
                'secret': BINANCE_TESTNET_SECRET,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future',
                    'adjustForTimeDifference': True,
                },
            })
            self.exchange.set_sandbox_mode(True)

            # Test connection
            balance = self.exchange.fetch_balance()
            usdt = balance.get('USDT', {}).get('free', 0)
            logger.info(f"Connected to Binance Futures Testnet | USDT Balance: {usdt}")
            self.connected = True
            return True

        except Exception as e:
            logger.error(f"Failed to connect to Binance Testnet: {e}")
            self.connected = False
            return False

    def get_balance(self):
        """Get USDT balance"""
        try:
            balance = self.exchange.fetch_balance()
            return float(balance.get('USDT', {}).get('free', 0))
        except Exception as e:
            logger.error(f"Balance fetch error: {e}")
            return 0

    def get_price(self, coin):
        """Get current price"""
        try:
            ticker = self.exchange.fetch_ticker(COINS[coin]['pair'])
            return ticker['last']
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

            # Calculate quantity (adjust for coin precision)
            markets = self.exchange.load_markets()
            market = self.exchange.market(pair)
            min_qty = market.get('limits', {}).get('amount', {}).get('min', 0.001)
            precision = market.get('precision', {}).get('amount', 3)

            quantity = round(position_value / price, precision)
            if quantity < min_qty:
                logger.warning(f"{coin}: Quantity {quantity} below minimum {min_qty}")
                return None

            # TP/SL prices
            tp_price = round(price * (1 + tp_pct), market.get('precision', {}).get('price', 2))
            sl_price = round(price * (1 - sl_pct), market.get('precision', {}).get('price', 2))

            logger.info(f"{coin}: Opening LONG | Price: {price} | Qty: {quantity} | TP: {tp_price} | SL: {sl_price}")

            # 1. Market buy order
            entry_order = self.exchange.create_order(
                symbol=pair,
                type='market',
                side='buy',
                amount=quantity,
            )

            entry_price = float(entry_order.get('average', price))
            logger.info(f"{coin}: Entry filled @ {entry_price}")

            # 2. Take Profit order (limit sell)
            try:
                tp_order = self.exchange.create_order(
                    symbol=pair,
                    type='TAKE_PROFIT_MARKET',
                    side='sell',
                    amount=quantity,
                    params={
                        'stopPrice': tp_price,
                        'closePosition': False,
                        'reduceOnly': True,
                    }
                )
                logger.info(f"{coin}: TP order placed @ {tp_price}")
            except Exception as e:
                logger.warning(f"{coin}: TP order failed ({e}), will monitor manually")
                tp_order = None

            # 3. Stop Loss order (stop market sell)
            try:
                sl_order = self.exchange.create_order(
                    symbol=pair,
                    type='STOP_MARKET',
                    side='sell',
                    amount=quantity,
                    params={
                        'stopPrice': sl_price,
                        'closePosition': False,
                        'reduceOnly': True,
                    }
                )
                logger.info(f"{coin}: SL order placed @ {sl_price}")
            except Exception as e:
                logger.warning(f"{coin}: SL order failed ({e}), will monitor manually")
                sl_order = None

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
