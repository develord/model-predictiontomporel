"""
Authentication Routes
======================
Google Sign-In + Binance OAuth2 + JWT token management
"""
import httpx
from fastapi import APIRouter, HTTPException, Depends, status
from datetime import datetime
import logging

from auth import (
    create_access_token,
    create_refresh_token,
    verify_token,
    get_current_user,
    ACCESS_TOKEN_EXPIRE_MINUTES,
)
from database import get_user_by_provider_id, create_user, update_last_login, initialize_credits
from models import (
    GoogleAuthRequest,
    BinanceAuthRequest,
    RefreshTokenRequest,
    AuthResponse,
    UserResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])

# Binance OAuth config (from env, set in config.py)
import os
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
BINANCE_CLIENT_ID = os.getenv("BINANCE_CLIENT_ID", "")
BINANCE_CLIENT_SECRET = os.getenv("BINANCE_CLIENT_SECRET", "")


def _build_auth_response(user: dict) -> dict:
    """Build JWT tokens and auth response for a user"""
    token_data = {
        "sub": str(user["id"]),
        "email": user.get("email"),
        "name": user.get("name"),
        "auth_provider": user["auth_provider"],
    }
    access_token = create_access_token(token_data)
    refresh_token = create_refresh_token(token_data)

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        "user": {
            "id": user["id"],
            "email": user.get("email"),
            "name": user.get("name"),
            "avatar": user.get("avatar"),
            "auth_provider": user["auth_provider"],
            "created_at": user.get("created_at", ""),
            "last_login": user.get("last_login", ""),
        }
    }


@router.post("/google", response_model=AuthResponse, summary="Login with Google")
async def auth_google(request: GoogleAuthRequest):
    """
    Authenticate with Google ID token.
    The app sends the id_token from Google Sign-In, server verifies it.
    """
    # Verify Google ID token
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"https://oauth2.googleapis.com/tokeninfo?id_token={request.id_token}",
            timeout=10.0,
        )

    if resp.status_code != 200:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Google ID token"
        )

    google_data = resp.json()

    # Verify audience matches our client ID
    if GOOGLE_CLIENT_ID and google_data.get("aud") != GOOGLE_CLIENT_ID:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token audience mismatch"
        )

    google_sub = google_data.get("sub")
    email = google_data.get("email")
    name = google_data.get("name", email)
    avatar = google_data.get("picture")

    if not google_sub:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Google token: missing sub"
        )

    # Find or create user
    user = await get_user_by_provider_id("google", google_sub)
    if user:
        await update_last_login(user["id"])
        user["last_login"] = datetime.utcnow().isoformat()
    else:
        user = await create_user(
            email=email,
            name=name,
            avatar=avatar,
            auth_provider="google",
            provider_user_id=google_sub,
        )
        await initialize_credits(user["id"], 3)
        logger.info(f"New Google user registered: {email} (+3 credits)")

    return _build_auth_response(user)


@router.post("/binance", response_model=AuthResponse, summary="Login with Binance")
async def auth_binance(request: BinanceAuthRequest):
    """
    Authenticate with Binance OAuth2 authorization code.
    The app opens Binance login in browser, gets code via deep link callback,
    then sends code here for exchange.
    """
    if not BINANCE_CLIENT_ID or not BINANCE_CLIENT_SECRET:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Binance OAuth not configured"
        )

    # Exchange authorization code for access token
    async with httpx.AsyncClient() as client:
        token_resp = await client.post(
            "https://accounts.binance.com/oauth/token",
            data={
                "grant_type": "authorization_code",
                "code": request.code,
                "client_id": BINANCE_CLIENT_ID,
                "client_secret": BINANCE_CLIENT_SECRET,
                "redirect_uri": request.redirect_uri,
            },
            timeout=15.0,
        )

    if token_resp.status_code != 200:
        logger.error(f"Binance token exchange failed: {token_resp.text}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Binance authentication failed"
        )

    binance_tokens = token_resp.json()
    binance_access_token = binance_tokens.get("access_token")

    if not binance_access_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Binance token response"
        )

    # Get Binance user info
    async with httpx.AsyncClient() as client:
        user_resp = await client.get(
            "https://accounts.binance.com/oauth-api/v1/user-info",
            headers={"Authorization": f"Bearer {binance_access_token}"},
            timeout=10.0,
        )

    if user_resp.status_code != 200:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Failed to get Binance user info"
        )

    binance_data = user_resp.json()
    binance_user_id = str(binance_data.get("userId", binance_data.get("id", "")))
    email = binance_data.get("email")
    name = binance_data.get("name", f"Binance User {binance_user_id[:8]}")

    if not binance_user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Binance user data"
        )

    # Find or create user
    user = await get_user_by_provider_id("binance", binance_user_id)
    if user:
        await update_last_login(user["id"])
        user["last_login"] = datetime.utcnow().isoformat()
    else:
        user = await create_user(
            email=email,
            name=name,
            avatar=None,
            auth_provider="binance",
            provider_user_id=binance_user_id,
        )
        await initialize_credits(user["id"], 3)
        logger.info(f"New Binance user registered: {binance_user_id} (+3 credits)")

    return _build_auth_response(user)


@router.post("/refresh", response_model=AuthResponse, summary="Refresh access token")
async def refresh_token(request: RefreshTokenRequest):
    """Exchange a valid refresh token for a new access token"""
    payload = verify_token(request.refresh_token, expected_type="refresh")

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )

    # Build new tokens with same user data
    user_data = {
        "id": int(user_id),
        "email": payload.get("email"),
        "name": payload.get("name"),
        "auth_provider": payload.get("auth_provider"),
        "avatar": None,
        "created_at": "",
        "last_login": datetime.utcnow().isoformat(),
    }

    return _build_auth_response(user_data)


@router.get("/me", response_model=UserResponse, summary="Get current user")
async def get_me(current_user: dict = Depends(get_current_user)):
    """Return the currently authenticated user info"""
    return {
        "id": current_user["id"],
        "email": current_user.get("email"),
        "name": current_user.get("name"),
        "avatar": current_user.get("avatar"),
        "auth_provider": current_user["auth_provider"],
        "created_at": current_user.get("created_at", ""),
        "last_login": current_user.get("last_login", ""),
    }
