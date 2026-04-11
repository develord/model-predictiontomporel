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
                type TEXT NOT NULL CHECK(type IN ('earn_ad', 'spend_view', 'bonus_signup')),
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


async def get_last_earn_time(user_id: int) -> str | None:
    """Get timestamp of last earn_ad transaction (for cooldown)"""
    async with aiosqlite.connect(DATABASE_PATH) as db:
        cursor = await db.execute(
            "SELECT timestamp FROM credit_transactions WHERE user_id = ? AND type = 'earn_ad' ORDER BY timestamp DESC LIMIT 1",
            (user_id,)
        )
        row = await cursor.fetchone()
        return row[0] if row else None
