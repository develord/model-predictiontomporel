"""
FCM Notification Service
=========================
Send push notifications via Firebase Cloud Messaging HTTP v1 API.
Uses service account JSON for auth (no firebase-admin SDK needed).
"""
import os
import json
import time
import logging
from pathlib import Path

import httpx
from jose import jwt as jose_jwt

logger = logging.getLogger(__name__)

# Path to Firebase service account JSON (downloaded from Firebase Console)
SERVICE_ACCOUNT_PATH = os.getenv(
    "FCM_SERVICE_ACCOUNT",
    str(Path(__file__).parent / "firebase-service-account.json")
)

# Cache for OAuth2 access token
_token_cache = {"token": None, "expires_at": 0}

COIN_NAMES = {
    'BTC': 'Bitcoin', 'ETH': 'Ethereum', 'SOL': 'Solana',
    'DOGE': 'Dogecoin', 'AVAX': 'Avalanche', 'XRP': 'XRP',
    'LINK': 'Chainlink', 'ADA': 'Cardano', 'NEAR': 'NEAR',
    'DOT': 'Polkadot', 'FIL': 'Filecoin',
}


def _load_service_account() -> dict | None:
    """Load Firebase service account credentials"""
    try:
        with open(SERVICE_ACCOUNT_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"FCM service account not found: {SERVICE_ACCOUNT_PATH}")
        return None


def _get_access_token() -> str | None:
    """Get a valid OAuth2 access token for FCM API (cached)"""
    now = time.time()
    if _token_cache["token"] and _token_cache["expires_at"] > now + 60:
        return _token_cache["token"]

    sa = _load_service_account()
    if not sa:
        return None

    # Build JWT for Google OAuth2
    iat = int(now)
    exp = iat + 3600
    payload = {
        "iss": sa["client_email"],
        "scope": "https://www.googleapis.com/auth/firebase.messaging",
        "aud": "https://oauth2.googleapis.com/token",
        "iat": iat,
        "exp": exp,
    }

    # Sign with service account private key (RS256)
    try:
        from jose import jwt as jose_jwt
        signed_jwt = jose_jwt.encode(payload, sa["private_key"], algorithm="RS256")
    except Exception as e:
        logger.error(f"JWT signing failed: {e}")
        return None

    # Exchange JWT for access token
    try:
        import httpx
        resp = httpx.post(
            "https://oauth2.googleapis.com/token",
            data={
                "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                "assertion": signed_jwt,
            },
            timeout=10.0,
        )
        if resp.status_code == 200:
            data = resp.json()
            _token_cache["token"] = data["access_token"]
            _token_cache["expires_at"] = now + data.get("expires_in", 3600)
            return _token_cache["token"]
        else:
            logger.error(f"OAuth2 token exchange failed: {resp.status_code} {resp.text}")
            return None
    except Exception as e:
        logger.error(f"OAuth2 token request failed: {e}")
        return None


async def send_signal_notification(
    tokens: list[str],
    coin: str,
    direction: str,
    confidence: float,
    price: float,
    tp_pct: float,
    sl_pct: float,
) -> tuple[int, list[str]]:
    """Send push notification to all registered devices.
    Returns (success_count, list_of_invalid_tokens)."""

    access_token = _get_access_token()
    if not access_token:
        logger.warning("No FCM access token, skipping push")
        return 0, []

    sa = _load_service_account()
    if not sa:
        return 0, []

    project_id = sa.get("project_id", "")
    url = f"https://fcm.googleapis.com/v1/projects/{project_id}/messages:send"

    coin_name = COIN_NAMES.get(coin, coin)
    emoji = "\U0001f7e2" if direction == "LONG" else "\U0001f534"  # green/red circle
    conf_pct = round(confidence * 100)

    title = f"{coin_name}: {direction} signal {conf_pct}% confidence"
    body = f"{emoji} {direction} @ ${price:,.2f} | TP: +{tp_pct:.1%} SL: -{sl_pct:.1%}"

    sent = 0
    failed_tokens = []

    async with httpx.AsyncClient() as client:
        for token in tokens:
            message = {
                "message": {
                    "token": token,
                    "notification": {
                        "title": title,
                        "body": body,
                    },
                    "data": {
                        "type": "signal",
                        "coin": coin,
                        "direction": direction,
                        "confidence": str(conf_pct),
                        "price": str(price),
                    },
                    "android": {
                        "priority": "high",
                        "notification": {
                            "channel_id": "signal-alerts",
                            "sound": "default",
                        },
                    },
                }
            }

            try:
                resp = await client.post(
                    url,
                    json=message,
                    headers={"Authorization": f"Bearer {access_token}"},
                    timeout=10.0,
                )
                if resp.status_code == 200:
                    sent += 1
                elif resp.status_code == 404 or "UNREGISTERED" in resp.text:
                    failed_tokens.append(token)
                    logger.info(f"FCM token invalid, removing: {token[:20]}...")
                else:
                    logger.warning(f"FCM send failed ({resp.status_code}): {resp.text[:200]}")
            except Exception as e:
                logger.error(f"FCM send error: {e}")

    return sent, failed_tokens
