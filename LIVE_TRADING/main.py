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

        # Check model is loaded
        if coin not in self.signal_gen.models:
            logger.warning(f"{coin}: No model loaded, skipping")
            return

        # Compute features
        feature_cols = self.signal_gen.feature_lists.get(coin, [])
        scaler = self.signal_gen.scalers.get(coin)
        if not feature_cols or scaler is None:
            logger.warning(f"{coin}: Missing features/scaler")
            return

        scaled_features, raw_row = compute_features(
            self.data_mgr.buffers, coin, feature_cols, scaler, SEQUENCE_LENGTH
        )

        if scaled_features is None:
            logger.warning(f"{coin}: Feature computation failed")
            return

        # Predict
        direction, confidence = self.signal_gen.predict(coin, scaled_features)
        if direction is None:
            return

        logger.info(f"{coin}: Prediction = {'LONG' if direction == 1 else 'SHORT'} | Confidence: {confidence:.1%}")

        # Only trade LONG signals
        if direction != 1:
            logger.info(f"{coin}: SHORT signal, no trade")
            return

        # Apply filters
        passes, reason = self.signal_gen.check_filters(coin, raw_row, confidence)
        if not passes:
            logger.info(f"{coin}: Filtered - {reason}")
            return

        # Calculate TP/SL
        current_price = self.executor.get_price(coin)
        if not current_price:
            return

        tp_pct, sl_pct = self.signal_gen.get_dynamic_tp_sl(raw_row, current_price)

        logger.info(f"{coin}: SIGNAL ACCEPTED | Conf: {confidence:.1%} | TP: {tp_pct:.2%} | SL: {sl_pct:.2%}")

        # Execute trade
        order_info = self.executor.open_long(coin, tp_pct, sl_pct)
        if order_info:
            self.pos_mgr.open_position(coin, order_info)
            logger.info(f"{coin}: TRADE EXECUTED | Entry: {order_info['entry_price']}")
        else:
            logger.error(f"{coin}: Trade execution failed")

    async def on_15m_candle_close(self, coin, candle):
        """Called on 15m candle close - check if position TP/SL hit (backup monitoring)"""
        if not self.pos_mgr.has_position(coin):
            return

        pos = self.pos_mgr.positions[coin]
        high = candle['high']
        low = candle['low']

        # Check if TP/SL was hit (backup - exchange orders should handle this)
        if high >= pos['tp_price']:
            logger.info(f"{coin}: TP hit detected on 15m candle (high={high}, tp={pos['tp_price']})")
            # Exchange TP order should have triggered, but verify
            positions = self.executor.get_open_positions()
            pair_symbol = COINS[coin]['pair'].replace('/', '')
            if pair_symbol not in positions or positions[pair_symbol]['contracts'] == 0:
                self.pos_mgr.close_position(coin, 'TP', pos['tp_price'])
                self.signal_gen.record_trade_result(coin, 'TP')

        elif low <= pos['sl_price']:
            logger.info(f"{coin}: SL hit detected on 15m candle (low={low}, sl={pos['sl_price']})")
            positions = self.executor.get_open_positions()
            pair_symbol = COINS[coin]['pair'].replace('/', '')
            if pair_symbol not in positions or positions[pair_symbol]['contracts'] == 0:
                self.pos_mgr.close_position(coin, 'SL', pos['sl_price'])
                self.signal_gen.record_trade_result(coin, 'SL')

    async def check_positions_sync(self):
        """Periodically sync local state with exchange positions"""
        while True:
            await asyncio.sleep(300)  # Every 5 minutes
            try:
                exchange_positions = self.executor.get_open_positions()

                for coin in list(self.pos_mgr.positions.keys()):
                    pair_symbol = COINS[coin]['pair'].replace('/', '')
                    if pair_symbol not in exchange_positions or exchange_positions[pair_symbol]['contracts'] == 0:
                        # Position was closed on exchange (TP/SL hit)
                        pos = self.pos_mgr.positions[coin]
                        current_price = self.executor.get_price(coin)
                        if current_price:
                            if current_price >= pos['tp_price']:
                                exit_type = 'TP'
                            elif current_price <= pos['sl_price']:
                                exit_type = 'SL'
                            else:
                                exit_type = 'CLOSED'
                            self.pos_mgr.close_position(coin, exit_type, current_price)
                            self.signal_gen.record_trade_result(coin, exit_type)
                            logger.info(f"{coin}: Position sync - closed ({exit_type})")

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
