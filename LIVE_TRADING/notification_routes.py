"""
Notification Routes
====================
FCM token registration + signal webhook (called by bot)
"""
import os
import hmac
import logging
from fastapi import APIRouter, HTTPException, Depends, Request, status
from pydantic import BaseModel

from auth import get_current_user
from database import save_device_token, get_all_device_tokens, remove_device_token, create_signal, close_signal, get_signal_history, get_signal_stats
from notification_service import send_signal_notification

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/notifications", tags=["Notifications"])

BOT_WEBHOOK_SECRET = os.getenv("BOT_WEBHOOK_SECRET", "")


class RegisterTokenRequest(BaseModel):
    fcm_token: str
    platform: str = "android"


class SignalWebhookPayload(BaseModel):
    coin: str
    direction: str  # LONG or SHORT
    confidence: float
    price: float
    tp_pct: float
    sl_pct: float


@router.post("/register", summary="Register FCM token for push notifications")
async def register_token(request: RegisterTokenRequest, current_user: dict = Depends(get_current_user)):
    """App calls this after login to register its FCM token."""
    if not request.fcm_token or len(request.fcm_token) < 20:
        raise HTTPException(status_code=400, detail="Invalid FCM token")

    await save_device_token(current_user["id"], request.fcm_token, request.platform)
    logger.info(f"FCM token registered for user {current_user['id']}")
    return {"status": "ok"}


@router.post("/signal-webhook", summary="Bot posts new signals here")
async def signal_webhook(payload: SignalWebhookPayload, request: Request):
    """Called by the live trading bot when a LONG/SHORT signal is accepted.
    Authenticated via X-Bot-Secret header (shared secret)."""
    bot_secret = request.headers.get("X-Bot-Secret", "")
    if not BOT_WEBHOOK_SECRET or not hmac.compare_digest(bot_secret, BOT_WEBHOOK_SECRET):
        raise HTTPException(status_code=403, detail="Forbidden")

    logger.info(f"Signal webhook: {payload.coin} {payload.direction} {payload.confidence:.1%}")

    # Persist signal to database
    signal_id = await create_signal(
        coin=payload.coin, direction=payload.direction,
        confidence=payload.confidence, entry_price=payload.price,
        tp_pct=payload.tp_pct, sl_pct=payload.sl_pct
    )
    logger.info(f"Signal stored: id={signal_id}")

    tokens = await get_all_device_tokens()
    if not tokens:
        logger.info("No device tokens registered, skipping push")
        return {"status": "ok", "sent": 0}

    fcm_tokens = [t["fcm_token"] for t in tokens]
    sent, failed = await send_signal_notification(
        tokens=fcm_tokens,
        coin=payload.coin,
        direction=payload.direction,
        confidence=payload.confidence,
        price=payload.price,
        tp_pct=payload.tp_pct,
        sl_pct=payload.sl_pct,
    )

    # Clean up invalid tokens
    for bad_token in failed:
        await remove_device_token(bad_token)

    logger.info(f"Push sent: {sent} success, {len(failed)} removed")
    return {"status": "ok", "sent": sent, "removed": len(failed)}


class CloseSignalPayload(BaseModel):
    coin: str
    direction: str  # LONG or SHORT
    result: str     # TP, SL, CLOSED
    exit_price: float
    pnl_pct: float


@router.post("/close-signal", summary="Bot posts trade close here")
async def close_signal_webhook(payload: CloseSignalPayload, request: Request):
    """Called by the live trading bot when a position is closed (TP/SL/timeout).
    Updates the stored signal with the result."""
    bot_secret = request.headers.get("X-Bot-Secret", "")
    if not BOT_WEBHOOK_SECRET or not hmac.compare_digest(bot_secret, BOT_WEBHOOK_SECRET):
        raise HTTPException(status_code=403, detail="Forbidden")

    signal_id = await close_signal(
        coin=payload.coin, direction=payload.direction,
        result=payload.result, exit_price=payload.exit_price,
        pnl_pct=payload.pnl_pct
    )
    logger.info(f"Signal closed: {payload.coin} {payload.direction} → {payload.result} ({payload.pnl_pct:+.2f}%) id={signal_id}")
    return {"status": "ok", "signal_id": signal_id}


@router.get("/history", summary="Get signal history")
async def signal_history(coin: str | None = None, limit: int = 100):
    """Return signal history + stats for the app. No auth required (public data)."""
    signals = await get_signal_history(coin=coin, limit=limit)
    stats = await get_signal_stats(coin=coin)
    return {"signals": signals, "stats": stats}
