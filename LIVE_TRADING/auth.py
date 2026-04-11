"""
Authentication - JWT + API Key
===============================
All secrets loaded from environment variables. Server refuses to start if missing.
"""
import os
import sys
import hmac
from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt

import logging

logger = logging.getLogger(__name__)

# Configuration — MUST be set via environment variables (no defaults)
SECRET_KEY = os.getenv("JWT_SECRET")
APP_API_KEY = os.getenv("APP_API_KEY")
TEST_TOKEN = os.getenv("TEST_TOKEN", "")  # Empty string = test token disabled

_missing = []
if not SECRET_KEY:
    _missing.append("JWT_SECRET")
if not APP_API_KEY:
    _missing.append("APP_API_KEY")
if _missing:
    logger.critical(f"FATAL: Missing required env vars: {', '.join(_missing)}")
    sys.exit(1)

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440  # 24 hours
REFRESH_TOKEN_EXPIRE_DAYS = 30

security = HTTPBearer()


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire, "type": "access"})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def create_refresh_token(data: dict) -> str:
    """Create a JWT refresh token (longer lived)"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh"})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def verify_token(token: str, expected_type: str = "access") -> dict:
    """Verify and decode a JWT token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("type") != expected_type:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid token type, expected {expected_type}"
            )
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )


def verify_api_key(api_key: str) -> bool:
    """Check if the API key matches the expected app key (timing-safe)"""
    return hmac.compare_digest(api_key, APP_API_KEY)


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """FastAPI dependency - extract and verify user from Bearer token.
    Supports test token bypass: Bearer test-crypto-2026-secret"""
    token = credentials.credentials
    # Test token bypass (only active if TEST_TOKEN env var is set)
    if TEST_TOKEN and hmac.compare_digest(token, TEST_TOKEN):
        return {"id": 0, "email": "test@test.com", "name": "Test User", "auth_provider": "test"}
    payload = verify_token(token, expected_type="access")
    user_id = payload.get("sub")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload"
        )
    return {
        "id": int(user_id),
        "email": payload.get("email"),
        "name": payload.get("name"),
        "auth_provider": payload.get("auth_provider"),
    }
