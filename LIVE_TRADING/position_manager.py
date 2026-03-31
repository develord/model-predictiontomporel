"""
Position Manager - Track open positions + state persistence
=============================================================
"""

import json
import logging
from pathlib import Path
from datetime import datetime

from config import STATE_DIR, TRADING

logger = logging.getLogger(__name__)

STATE_FILE = STATE_DIR / 'positions.json'
TRADE_LOG = STATE_DIR / 'trade_log.json'


class PositionManager:
    def __init__(self):
        self.positions = {}  # {coin: position_info}
        self.trade_history = []
        self._load_state()

    def _load_state(self):
        """Load persisted state on startup"""
        try:
            if STATE_FILE.exists():
                with open(STATE_FILE) as f:
                    self.positions = json.load(f)
                logger.info(f"Loaded {len(self.positions)} open positions from state")
        except Exception as e:
            logger.warning(f"Could not load state: {e}")
            self.positions = {}

        try:
            if TRADE_LOG.exists():
                with open(TRADE_LOG) as f:
                    self.trade_history = json.load(f)
        except:
            self.trade_history = []

    def _save_state(self):
        """Persist current positions"""
        try:
            STATE_DIR.mkdir(parents=True, exist_ok=True)
            with open(STATE_FILE, 'w') as f:
                json.dump(self.positions, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def _save_trade(self, trade):
        """Append trade to log"""
        try:
            self.trade_history.append(trade)
            with open(TRADE_LOG, 'w') as f:
                json.dump(self.trade_history, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save trade log: {e}")

    def has_position(self, coin):
        return coin in self.positions

    def open_position(self, coin, order_info):
        """Record a new open position"""
        self.positions[coin] = {
            'coin': coin,
            'direction': order_info.get('direction', order_info.get('side', 'LONG')),
            'entry_price': order_info['entry_price'],
            'quantity': order_info['quantity'],
            'tp_price': order_info['tp_price'],
            'sl_price': order_info['sl_price'],
            'tp_pct': order_info['tp_pct'],
            'sl_pct': order_info['sl_pct'],
            'entry_time': datetime.utcnow().isoformat(),
            'entry_order_id': order_info.get('entry_order_id'),
            'tp_order_id': order_info.get('tp_order_id'),
            'sl_order_id': order_info.get('sl_order_id'),
        }
        self._save_state()
        logger.info(f"{coin}: Position opened @ {order_info['entry_price']} | TP: {order_info['tp_price']} | SL: {order_info['sl_price']}")

    def close_position(self, coin, exit_type, exit_price):
        """Record position close"""
        if coin not in self.positions:
            return

        pos = self.positions[coin]
        pnl_pct = (exit_price / pos['entry_price'] - 1) * 100

        trade = {
            'coin': coin,
            'entry_price': pos['entry_price'],
            'exit_price': exit_price,
            'entry_time': pos['entry_time'],
            'exit_time': datetime.utcnow().isoformat(),
            'exit_type': exit_type,
            'pnl_pct': round(pnl_pct, 2),
            'quantity': pos['quantity'],
        }
        self._save_trade(trade)

        logger.info(f"{coin}: Position closed ({exit_type}) @ {exit_price} | PnL: {pnl_pct:+.2f}%")

        del self.positions[coin]
        self._save_state()

        return exit_type

    def get_summary(self):
        """Get portfolio summary"""
        total_trades = len(self.trade_history)
        if total_trades == 0:
            return "No trades yet"

        wins = sum(1 for t in self.trade_history if t['pnl_pct'] > 0)
        total_pnl = sum(t['pnl_pct'] for t in self.trade_history)
        wr = wins / total_trades * 100

        summary = f"Trades: {total_trades} | Wins: {wins} | WR: {wr:.0f}% | Total PnL: {total_pnl:+.1f}%"
        if self.positions:
            summary += f" | Open: {list(self.positions.keys())}"
        return summary
