"""
Live Trading System - Main Orchestrator
=========================================
Connects all modules: data -> features -> signals -> trades

Usage:
    python main.py

Environment variables required:
    BINANCE_TESTNET_KEY=your_testnet_api_key
    BINANCE_TESTNET_SECRET=your_testnet_api_secret
"""

import asyncio
import logging
import sys
from pathlib import Path
from datetime import datetime

# Setup logging
LOG_DIR = Path(__file__).parent / 'logs'
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / f'live_{datetime.now().strftime("%Y%m%d")}.log'),
    ]
)
logger = logging.getLogger('LIVE')

from config import COINS, SEQUENCE_LENGTH
from data_manager import DataManager
from feature_engine import compute_features
from signal_generator import SignalGenerator
from trade_executor import TradeExecutor
from position_manager import PositionManager


class LiveTradingSystem:
    def __init__(self):
        self.data_mgr = DataManager()
        self.signal_gen = SignalGenerator()
        self.executor = TradeExecutor()
        self.pos_mgr = PositionManager()

    def setup(self):
        """Initialize all components"""
        logger.info("=" * 70)
        logger.info("LIVE TRADING SYSTEM - STARTUP")
        logger.info("=" * 70)

        # Load models
        logger.info("\n[1/3] Loading CNN models...")
        self.signal_gen.load_models()

        # Connect to exchange
        logger.info("\n[2/3] Connecting to Binance Futures Testnet...")
        if not self.executor.connect():
            logger.error("Failed to connect to exchange. Check API keys.")
            return False

        # Bootstrap data
        logger.info("\n[3/3] Bootstrapping historical data...")
        self.data_mgr.bootstrap()

        # Register callbacks
        self.data_mgr.on_daily_close = self.on_daily_candle_close
        self.data_mgr.on_15m_close = self.on_15m_candle_close

        logger.info("\n" + "=" * 70)
        logger.info("SYSTEM READY - Waiting for signals")
        logger.info("=" * 70)
        logger.info(f"Coins: {', '.join(COINS.keys())}")
        logger.info(f"Open positions: {list(self.pos_mgr.positions.keys()) or 'None'}")
        logger.info(f"Balance: ${self.executor.get_balance():.2f}")
        logger.info(f"{self.pos_mgr.get_summary()}")

        return True

    async def on_daily_candle_close(self, coin):
        """Called when a 1d candle closes - generate signal and trade"""
        logger.info(f"\n{'='*50}")
        logger.info(f"DAILY CLOSE: {coin} @ {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")

        # Skip if already in position
        if self.pos_mgr.has_position(coin):
            logger.info(f"{coin}: Already in position, skipping")
            return

        # Compute LONG features
        long_feat_cols = self.signal_gen.long_features.get(coin, [])
        long_scaler = self.signal_gen.long_scalers.get(coin)

        raw_row = None
        long_scaled = None
        if long_feat_cols and long_scaler:
            long_scaled, raw_row = compute_features(
                self.data_mgr.buffers, coin, long_feat_cols, long_scaler, SEQUENCE_LENGTH
            )

        # Compute SHORT features (may need different seq_len)
        short_feat_cols = self.signal_gen.short_features.get(coin, [])
        short_scaler = self.signal_gen.short_scalers.get(coin)
        short_seq = getattr(self.signal_gen, 'short_seq_lens', {}).get(coin, SEQUENCE_LENGTH)

        short_scaled = None
        if short_feat_cols and short_scaler:
            short_scaled, raw_row_s = compute_features(
                self.data_mgr.buffers, coin, short_feat_cols, short_scaler, short_seq
            )
            if raw_row is None:
                raw_row = raw_row_s

        if raw_row is None:
            logger.warning(f"{coin}: Feature computation failed")
            return

        current_price = self.executor.get_price(coin)
        if not current_price:
            return

        # Try LONG first
        if long_scaled is not None and coin in self.signal_gen.long_models:
            l_dir, l_conf = self.signal_gen.predict_long(coin, long_scaled)
            if l_dir == 1:
                logger.info(f"{coin}: LONG signal | Conf: {l_conf:.1%}")
                passes, reason = self.signal_gen.check_long_filters(coin, raw_row, l_conf)
                if passes:
                    tp_pct, sl_pct = self.signal_gen.get_dynamic_tp_sl(raw_row, current_price, 'LONG')
                    logger.info(f"{coin}: LONG ACCEPTED | TP: {tp_pct:.2%} | SL: {sl_pct:.2%}")
                    order = self.executor.open_long(coin, tp_pct, sl_pct)
                    if order:
                        order['direction'] = 'LONG'
                        self.pos_mgr.open_position(coin, order)
                        return
                else:
                    logger.info(f"{coin}: LONG filtered - {reason}")

        # Try SHORT if LONG didn't trigger
        if short_scaled is not None and coin in self.signal_gen.short_models:
            s_dir, s_conf = self.signal_gen.predict_short(coin, short_scaled)
            if s_dir == 1:
                logger.info(f"{coin}: SHORT signal | Conf: {s_conf:.1%}")
                passes, reason = self.signal_gen.check_short_filters(coin, raw_row, s_conf)
                if passes:
                    tp_pct, sl_pct = self.signal_gen.get_dynamic_tp_sl(raw_row, current_price, 'SHORT')
                    logger.info(f"{coin}: SHORT ACCEPTED | TP: {tp_pct:.2%} | SL: {sl_pct:.2%}")
                    order = self.executor.open_short(coin, tp_pct, sl_pct)
                    if order:
                        order['direction'] = 'SHORT'
                        self.pos_mgr.open_position(coin, order)
                        return
                else:
                    logger.info(f"{coin}: SHORT filtered - {reason}")

        logger.info(f"{coin}: No trade signal")

    async def on_15m_candle_close(self, coin, candle):
        """Called on 15m candle close - monitor SL (exchange can't do STOP on demo)"""
        if not self.pos_mgr.has_position(coin):
            return

        pos = self.pos_mgr.positions[coin]
        high = candle['high']
        low = candle['low']
        direction = pos.get('direction', pos.get('side', 'LONG'))

        if direction == 'LONG':
            # LONG: TP when high >= tp_price, SL when low <= sl_price
            if low <= pos['sl_price']:
                logger.info(f"{coin}: LONG SL HIT (low={low} <= sl={pos['sl_price']})")
                self.executor.close_position(coin, pos['quantity'], 'LONG')
                self.executor.cancel_orders(coin)
                self.pos_mgr.close_position(coin, 'SL', pos['sl_price'])
                self.signal_gen.record_trade_result(coin, 'LONG', 'SL')
            elif high >= pos['tp_price']:
                logger.info(f"{coin}: LONG TP HIT (high={high} >= tp={pos['tp_price']})")
                # TP order on exchange should have filled, verify
                self.pos_mgr.close_position(coin, 'TP', pos['tp_price'])
                self.signal_gen.record_trade_result(coin, 'LONG', 'TP')
        else:
            # SHORT: TP when low <= tp_price (price drops), SL when high >= sl_price (price rises)
            if high >= pos['sl_price']:
                logger.info(f"{coin}: SHORT SL HIT (high={high} >= sl={pos['sl_price']})")
                self.executor.close_position(coin, pos['quantity'], 'SHORT')
                self.executor.cancel_orders(coin)
                self.pos_mgr.close_position(coin, 'SL', pos['sl_price'])
                self.signal_gen.record_trade_result(coin, 'SHORT', 'SL')
            elif low <= pos['tp_price']:
                logger.info(f"{coin}: SHORT TP HIT (low={low} <= tp={pos['tp_price']})")
                # TP order on exchange should have filled, verify
                self.pos_mgr.close_position(coin, 'TP', pos['tp_price'])
                self.signal_gen.record_trade_result(coin, 'SHORT', 'TP')

    async def check_positions_sync(self):
        """Periodically sync local state with exchange positions"""
        while True:
            await asyncio.sleep(300)  # Every 5 minutes
            try:
                exchange_positions = self.executor.get_open_positions()

                for coin in list(self.pos_mgr.positions.keys()):
                    pair_symbol = COINS[coin]['pair'].replace('/', '')  # BTC/USDT -> BTCUSDT
                    found = pair_symbol in exchange_positions
                    if not found:
                        pos = self.pos_mgr.positions[coin]
                        direction = pos.get('direction', 'LONG')
                        current_price = self.executor.get_price(coin)
                        if current_price and current_price > 0:
                            tp = pos.get('tp_price', 0)
                            sl = pos.get('sl_price', 0)
                            if tp > 0 and sl > 0:
                                if direction == 'SHORT':
                                    exit_type = 'TP' if current_price <= tp else ('SL' if current_price >= sl else 'CLOSED')
                                else:
                                    exit_type = 'TP' if current_price >= tp else ('SL' if current_price <= sl else 'CLOSED')
                                self.pos_mgr.close_position(coin, exit_type, current_price)
                                self.signal_gen.record_trade_result(coin, direction, exit_type)
                                logger.info(f"{coin}: Sync - {direction} closed ({exit_type}) @ {current_price}")

            except Exception as e:
                logger.error(f"Position sync error: {e}")

    async def status_loop(self):
        """Print status every minute"""
        while True:
            await asyncio.sleep(60)
            logger.info(f"[STATUS] {datetime.utcnow().strftime('%H:%M UTC')} | {self.pos_mgr.get_summary()}")

    async def run(self):
        """Main event loop"""
        if not self.setup():
            return

        # Run immediate prediction on startup (for testing)
        logger.info("\n--- Running predictions on current data ---")
        for coin in COINS:
            await self.on_daily_candle_close(coin)

        logger.info("\n--- Starting WebSocket stream ---")

        # Run concurrent tasks
        await asyncio.gather(
            self.data_mgr.run_websocket(),
            self.check_positions_sync(),
            self.status_loop(),
        )


def main():
    system = LiveTradingSystem()
    try:
        asyncio.run(system.run())
    except KeyboardInterrupt:
        logger.info("\nShutting down...")
        system.data_mgr.stop()


if __name__ == "__main__":
    main()
