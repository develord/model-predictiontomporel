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
        """Called on every 15m tick - trailing stop + TP/SL monitoring (real-time)"""
        if not self.pos_mgr.has_position(coin):
            return

        pos = self.pos_mgr.positions[coin]
        high = candle['high']
        low = candle['low']
        close = candle['close']
        direction = pos.get('direction', pos.get('side', 'LONG'))
        entry = pos.get('entry_price', 0)
        tp_price = pos.get('tp_price', 0)
        sl_price = pos.get('sl_price', 0)

        if entry <= 0:
            return

        # Cooldown: don't update trailing stop more than once per minute
        import time
        last_trail = pos.get('_last_trail_check', 0)
        now = time.time()
        if now - last_trail < 60:
            # Still check TP/SL hit (critical) but skip trailing logic
            pass_trailing = True
        else:
            pass_trailing = False
            pos['_last_trail_check'] = now

        # Calculate current PnL %
        if direction == 'LONG':
            current_pnl_pct = (close / entry - 1) * 100
            tp_total_pct = (tp_price / entry - 1) * 100 if tp_price > 0 else 0
        else:
            current_pnl_pct = (entry / close - 1) * 100
            tp_total_pct = (entry / tp_price - 1) * 100 if tp_price > 0 else 0

        # ====== TRAILING STOP LOGIC (every ~60s) ======
        if not pass_trailing:
            # 1. Close at 90% of TP target
            if tp_total_pct > 0 and current_pnl_pct >= tp_total_pct * 0.90:
                logger.info(f"{coin}: PROXIMITY CLOSE | PnL: {current_pnl_pct:+.2f}% (90% of TP {tp_total_pct:.1f}%)")
                self.executor.close_position(coin, pos['quantity'], direction)
                self.executor.cancel_orders(coin)
                self.pos_mgr.close_position(coin, 'TP_TRAIL', close)
                self.signal_gen.record_trade_result(coin, direction, 'TP')
                return

            # 2. Trailing SL adjustments
            new_sl = sl_price
            trail_reason = None

            if current_pnl_pct >= 3.0:
                if direction == 'LONG':
                    new_sl = max(sl_price, entry * 1.02)
                else:
                    new_sl = min(sl_price, entry * 0.98)
                trail_reason = "trail_3pct_lock_2pct"
            elif current_pnl_pct >= 2.5:
                if direction == 'LONG':
                    new_sl = max(sl_price, entry * 1.01)
                else:
                    new_sl = min(sl_price, entry * 0.99)
                trail_reason = "trail_2.5pct_lock_1pct"
            elif current_pnl_pct >= 1.5:
                if direction == 'LONG':
                    new_sl = max(sl_price, entry)
                else:
                    new_sl = min(sl_price, entry)
                trail_reason = "trail_1.5pct_breakeven"

            # Apply trailing SL if changed
            if trail_reason:
                sl_changed = (direction == 'LONG' and new_sl > sl_price) or \
                             (direction == 'SHORT' and new_sl < sl_price)
                if sl_changed:
                    logger.info(f"{coin}: TRAILING STOP | {trail_reason} | PnL: {current_pnl_pct:+.2f}% | SL: {sl_price:.4f} -> {new_sl:.4f}")
                    from trade_executor import round_to_tick
                    try:
                        self.executor.cancel_orders(coin)
                        info = self.executor.exchange.fapiPublicGetExchangeInfo()
                        sym = COINS[coin]['pair'].replace('/', '')
                        si = next((s for s in info['symbols'] if s['symbol'] == sym), None)
                        tick = next((f['tickSize'] for f in si['filters'] if f['filterType'] == 'PRICE_FILTER'), '0.01') if si else '0.01'
                        new_sl_rounded = round_to_tick(new_sl, tick)
                        tp_rounded = round_to_tick(tp_price, tick)

                        tp_side = 'SELL' if direction == 'LONG' else 'BUY'
                        self.executor.exchange.fapiPrivatePostOrder({
                            'symbol': sym, 'side': tp_side, 'type': 'LIMIT',
                            'price': tp_rounded, 'quantity': pos['quantity'],
                            'timeInForce': 'GTC', 'reduceOnly': 'true',
                        })
                        sl_side = 'SELL' if direction == 'LONG' else 'BUY'
                        self.executor._place_algo_order(sym, sl_side, 'STOP_MARKET', new_sl_rounded, pos['quantity'])

                        pos['sl_price'] = new_sl_rounded
                        self.pos_mgr._save_state()
                        logger.info(f"{coin}: New SL @ {new_sl_rounded} + TP @ {tp_rounded}")
                    except Exception as e:
                        logger.error(f"{coin}: Trailing update failed: {e}")

        # ====== SL/TP HIT CHECK (backup) ======

        if direction == 'LONG':
            if low <= sl_price:
                logger.info(f"{coin}: LONG SL HIT (low={low} <= sl={sl_price})")
                self.executor.close_position(coin, pos['quantity'], 'LONG')
                self.executor.cancel_orders(coin)
                self.pos_mgr.close_position(coin, 'SL', sl_price)
                self.signal_gen.record_trade_result(coin, 'LONG', 'SL')
            elif high >= tp_price:
                logger.info(f"{coin}: LONG TP HIT")
                self.pos_mgr.close_position(coin, 'TP', tp_price)
                self.signal_gen.record_trade_result(coin, 'LONG', 'TP')
        else:
            if high >= sl_price:
                logger.info(f"{coin}: SHORT SL HIT (high={high} >= sl={sl_price})")
                self.executor.close_position(coin, pos['quantity'], 'SHORT')
                self.executor.cancel_orders(coin)
                self.pos_mgr.close_position(coin, 'SL', sl_price)
                self.signal_gen.record_trade_result(coin, 'SHORT', 'SL')
            elif low <= tp_price:
                logger.info(f"{coin}: SHORT TP HIT")
                self.pos_mgr.close_position(coin, 'TP', tp_price)
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
