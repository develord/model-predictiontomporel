"""
Database Layer - SQLite User Management
========================================
"""
import aiosqlite
import os
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

DATABASE_PATH = os.getenv(
    "DATABASE_PATH",
    str(Path(__file__).parent / "data" / "users.db")
)


async def init_db():
    """Initialize SQLite database and create tables"""
    os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
    async with aiosqlite.connect(DATABASE_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT,
                name TEXT,
                avatar TEXT,
                auth_provider TEXT NOT NULL CHECK(auth_provider IN ('google', 'binance')),
                provider_user_id TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                last_login TEXT NOT NULL DEFAULT (datetime('now')),
                UNIQUE(auth_provider, provider_user_id)
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS credits (
                user_id INTEGER PRIMARY KEY REFERENCES users(id),
                balance INTEGER NOT NULL DEFAULT 0,
                last_updated TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS credit_transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id),
                amount INTEGER NOT NULL,
                type TEXT NOT NULL CHECK(type IN ('earn_ad', 'earn_ad_ssv', 'spend_view', 'bonus_signup')),
                crypto TEXT,
                timestamp TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS device_tokens (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id),
                fcm_token TEXT NOT NULL,
                platform TEXT NOT NULL DEFAULT 'android',
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                UNIQUE(user_id, fcm_token)
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                coin TEXT NOT NULL,
                direction TEXT NOT NULL,
                confidence REAL NOT NULL,
                entry_price REAL NOT NULL,
                tp_pct REAL NOT NULL,
                sl_pct REAL NOT NULL,
                result TEXT,
                exit_price REAL,
                pnl_pct REAL,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                closed_at TEXT
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS ssv_rewards (
                transaction_id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                amount INTEGER NOT NULL,
                processed_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)

        # Migration: recreate credit_transactions if CHECK constraint is outdated
        # (SQLite doesn't support ALTER TABLE to modify constraints)
        cursor = await db.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='credit_transactions'"
        )
        row = await cursor.fetchone()
        if row and row[0] and 'earn_ad_ssv' not in row[0]:
            logger.info("Migrating credit_transactions table to add earn_ad_ssv type...")
            await db.execute("ALTER TABLE credit_transactions RENAME TO credit_transactions_old")
            await db.execute("""
                CREATE TABLE credit_transactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL REFERENCES users(id),
                    amount INTEGER NOT NULL,
                    type TEXT NOT NULL CHECK(type IN ('earn_ad', 'earn_ad_ssv', 'spend_view', 'bonus_signup')),
                    crypto TEXT,
                    timestamp TEXT NOT NULL DEFAULT (datetime('now'))
                )
            """)
            await db.execute("""
                INSERT INTO credit_transactions (id, user_id, amount, type, crypto, timestamp)
                SELECT id, user_id, amount, type, crypto, timestamp FROM credit_transactions_old
            """)
            await db.execute("DROP TABLE credit_transactions_old")
            logger.info("Migration complete: credit_transactions updated")

        await db.commit()
    logger.info(f"Database initialized at {DATABASE_PATH}")


async def get_user_by_provider_id(auth_provider: str, provider_user_id: str) -> dict | None:
    """Find user by auth provider and provider user ID"""
    async with aiosqlite.connect(DATABASE_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM users WHERE auth_provider = ? AND provider_user_id = ?",
            (auth_provider, provider_user_id)
        )
        row = await cursor.fetchone()
        if row:
            return dict(row)
        return None


async def create_user(email: str, name: str, avatar: str, auth_provider: str, provider_user_id: str) -> dict:
    """Create a new user and return it"""
    now = datetime.utcnow().isoformat()
    async with aiosqlite.connect(DATABASE_PATH) as db:
        cursor = await db.execute(
            """INSERT INTO users (email, name, avatar, auth_provider, provider_user_id, created_at, last_login)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (email, name, avatar, auth_provider, provider_user_id, now, now)
        )
        await db.commit()
        user_id = cursor.lastrowid

    return {
        "id": user_id,
        "email": email,
        "name": name,
        "avatar": avatar,
        "auth_provider": auth_provider,
        "provider_user_id": provider_user_id,
        "created_at": now,
        "last_login": now,
    }


async def update_last_login(user_id: int):
    """Update user's last login timestamp"""
    now = datetime.utcnow().isoformat()
    async with aiosqlite.connect(DATABASE_PATH) as db:
        await db.execute(
            "UPDATE users SET last_login = ? WHERE id = ?",
            (now, user_id)
        )
        await db.commit()


# ============================================================================
# CREDITS
# ============================================================================

async def get_credits(user_id: int) -> dict:
    """Get user's credit balance"""
    async with aiosqlite.connect(DATABASE_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT balance, last_updated FROM credits WHERE user_id = ?",
            (user_id,)
        )
        row = await cursor.fetchone()
        if row:
            return {"balance": row["balance"], "last_updated": row["last_updated"]}
        return {"balance": 0, "last_updated": datetime.utcnow().isoformat()}


async def initialize_credits(user_id: int, amount: int = 50) -> int:
    """Give initial credits to a new user (idempotent)"""
    now = datetime.utcnow().isoformat()
    async with aiosqlite.connect(DATABASE_PATH) as db:
        await db.execute(
            "INSERT OR IGNORE INTO credits (user_id, balance, last_updated) VALUES (?, ?, ?)",
            (user_id, amount, now)
        )
        if amount > 0:
            await db.execute(
                "INSERT INTO credit_transactions (user_id, amount, type, timestamp) VALUES (?, ?, 'bonus_signup', ?)",
                (user_id, amount, now)
            )
        await db.commit()
    return amount


async def add_credits(user_id: int, amount: int, tx_type: str, crypto: str = None) -> int:
    """Add credits to user balance (for ad rewards)"""
    now = datetime.utcnow().isoformat()
    async with aiosqlite.connect(DATABASE_PATH) as db:
        # Ensure credits row exists
        await db.execute(
            "INSERT OR IGNORE INTO credits (user_id, balance, last_updated) VALUES (?, 0, ?)",
            (user_id, now)
        )
        await db.execute(
            "UPDATE credits SET balance = balance + ?, last_updated = ? WHERE user_id = ?",
            (amount, now, user_id)
        )
        await db.execute(
            "INSERT INTO credit_transactions (user_id, amount, type, crypto, timestamp) VALUES (?, ?, ?, ?, ?)",
            (user_id, amount, tx_type, crypto, now)
        )
        await db.commit()
        cursor = await db.execute("SELECT balance FROM credits WHERE user_id = ?", (user_id,))
        row = await cursor.fetchone()
        return row[0] if row else 0


async def spend_credits(user_id: int, amount: int, crypto: str) -> int | None:
    """Spend credits to view a prediction. Returns new balance or None if insufficient.
    Uses atomic UPDATE with WHERE balance >= amount to prevent race conditions."""
    now = datetime.utcnow().isoformat()
    async with aiosqlite.connect(DATABASE_PATH) as db:
        # Atomic: only deduct if balance is sufficient (prevents double-spend)
        cursor = await db.execute(
            "UPDATE credits SET balance = balance - ?, last_updated = ? WHERE user_id = ? AND balance >= ?",
            (amount, now, user_id, amount)
        )
        if cursor.rowcount == 0:
            return None

        await db.execute(
            "INSERT INTO credit_transactions (user_id, amount, type, crypto, timestamp) VALUES (?, ?, 'spend_view', ?, ?)",
            (user_id, -amount, crypto, now)
        )
        await db.commit()
        row_cursor = await db.execute("SELECT balance FROM credits WHERE user_id = ?", (user_id,))
        row = await row_cursor.fetchone()
        return row[0] if row else 0


async def save_device_token(user_id: int, fcm_token: str, platform: str = 'android'):
    """Save or update a device FCM token for push notifications"""
    now = datetime.utcnow().isoformat()
    async with aiosqlite.connect(DATABASE_PATH) as db:
        await db.execute(
            """INSERT INTO device_tokens (user_id, fcm_token, platform, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(user_id, fcm_token) DO UPDATE SET updated_at = ?""",
            (user_id, fcm_token, platform, now, now, now)
        )
        await db.commit()


async def remove_device_token(fcm_token: str):
    """Remove an invalid/expired FCM token"""
    async with aiosqlite.connect(DATABASE_PATH) as db:
        await db.execute("DELETE FROM device_tokens WHERE fcm_token = ?", (fcm_token,))
        await db.commit()


async def get_all_device_tokens() -> list[dict]:
    """Get all registered device tokens for broadcasting"""
    async with aiosqlite.connect(DATABASE_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT user_id, fcm_token, platform FROM device_tokens")
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]


async def get_ssv_reward(transaction_id: str) -> dict | None:
    """Check if an SSV transaction was already processed (dedup)"""
    if not transaction_id:
        return None
    async with aiosqlite.connect(DATABASE_PATH) as db:
        cursor = await db.execute(
            "SELECT transaction_id, user_id, amount FROM ssv_rewards WHERE transaction_id = ?",
            (transaction_id,)
        )
        row = await cursor.fetchone()
        if row:
            return {"transaction_id": row[0], "user_id": row[1], "amount": row[2]}
        return None


async def record_ssv_reward(transaction_id: str, user_id: int, amount: int):
    """Record an SSV transaction to prevent duplicates"""
    if not transaction_id:
        return
    now = datetime.utcnow().isoformat()
    async with aiosqlite.connect(DATABASE_PATH) as db:
        await db.execute(
            "INSERT OR IGNORE INTO ssv_rewards (transaction_id, user_id, amount, processed_at) VALUES (?, ?, ?, ?)",
            (transaction_id, user_id, amount, now)
        )
        await db.commit()


async def get_last_ssv_time(user_id: int) -> str | None:
    """Get timestamp of last SSV reward for a user (for dedup with client-side earn)"""
    async with aiosqlite.connect(DATABASE_PATH) as db:
        cursor = await db.execute(
            "SELECT processed_at FROM ssv_rewards WHERE user_id = ? ORDER BY processed_at DESC LIMIT 1",
            (user_id,)
        )
        row = await cursor.fetchone()
        return row[0] if row else None


async def get_last_earn_time(user_id: int) -> str | None:
    """Get timestamp of last earn_ad transaction (for cooldown)"""
    async with aiosqlite.connect(DATABASE_PATH) as db:
        cursor = await db.execute(
            "SELECT timestamp FROM credit_transactions WHERE user_id = ? AND type = 'earn_ad' ORDER BY timestamp DESC LIMIT 1",
            (user_id,)
        )
        row = await cursor.fetchone()
        return row[0] if row else None


# ============================================================
# SIGNALS
# ============================================================

async def create_signal(coin: str, direction: str, confidence: float,
                        entry_price: float, tp_pct: float, sl_pct: float) -> int:
    """Store a new signal, return its ID"""
    async with aiosqlite.connect(DATABASE_PATH) as db:
        cursor = await db.execute(
            """INSERT INTO signals (coin, direction, confidence, entry_price, tp_pct, sl_pct, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (coin, direction, confidence, entry_price, tp_pct, sl_pct, datetime.utcnow().isoformat())
        )
        await db.commit()
        return cursor.lastrowid


async def close_signal(coin: str, direction: str, result: str,
                       exit_price: float, pnl_pct: float):
    """Close the most recent open signal for a coin+direction"""
    now = datetime.utcnow().isoformat()
    async with aiosqlite.connect(DATABASE_PATH) as db:
        # Find the latest open signal for this coin+direction
        cursor = await db.execute(
            """SELECT id FROM signals
               WHERE coin = ? AND direction = ? AND result IS NULL
               ORDER BY created_at DESC LIMIT 1""",
            (coin, direction)
        )
        row = await cursor.fetchone()
        if row:
            await db.execute(
                """UPDATE signals SET result = ?, exit_price = ?, pnl_pct = ?, closed_at = ?
                   WHERE id = ?""",
                (result, exit_price, pnl_pct, now, row[0])
            )
            await db.commit()
            return row[0]
    return None


async def get_signal_history(coin: str | None = None, limit: int = 100) -> list[dict]:
    """Get signal history, newest first. Optionally filter by coin."""
    async with aiosqlite.connect(DATABASE_PATH) as db:
        db.row_factory = aiosqlite.Row
        if coin:
            cursor = await db.execute(
                """SELECT id, coin, direction, confidence, entry_price as price,
                          tp_pct, sl_pct, result, exit_price, pnl_pct,
                          created_at, closed_at
                   FROM signals WHERE coin = ? ORDER BY created_at DESC LIMIT ?""",
                (coin, limit)
            )
        else:
            cursor = await db.execute(
                """SELECT id, coin, direction, confidence, entry_price as price,
                          tp_pct, sl_pct, result, exit_price, pnl_pct,
                          created_at, closed_at
                   FROM signals ORDER BY created_at DESC LIMIT ?""",
                (limit,)
            )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]


async def get_signal_stats(coin: str | None = None) -> dict:
    """Compute signal stats (wins, losses, pending, win_rate, avg_pnl)"""
    signals = await get_signal_history(coin, limit=10000)
    total = len(signals)
    wins = sum(1 for s in signals if s['result'] == 'TP')
    losses = sum(1 for s in signals if s['result'] == 'SL')
    pending = sum(1 for s in signals if s['result'] is None)
    closed = [s for s in signals if s['pnl_pct'] is not None]
    avg_pnl = round(sum(s['pnl_pct'] for s in closed) / len(closed), 2) if closed else 0
    win_rate = round(wins / (wins + losses) * 100) if (wins + losses) > 0 else 0
    return {
        'total': total, 'wins': wins, 'losses': losses,
        'pending': pending, 'avg_pnl': avg_pnl, 'win_rate': win_rate
    }
