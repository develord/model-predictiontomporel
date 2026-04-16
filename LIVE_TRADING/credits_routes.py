"""
Credits Routes
===============
Earn credits (ad rewards) + spend credits (view predictions)
Includes AdMob SSV (Server-Side Verification) callback
"""
from fastapi import APIRouter, HTTPException, Depends, Request, status
from datetime import datetime
import logging
import hashlib
import hmac
import urllib.parse

from auth import get_current_user
from database import get_credits, add_credits, spend_credits, get_last_earn_time, get_ssv_reward, record_ssv_reward, get_last_ssv_time
from models import CreditsResponse, EarnCreditsRequest, SpendCreditsRequest, SpendCreditsResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/credits", tags=["Credits"])

EARN_COOLDOWN_SECONDS = 10  # Min time between ad rewards


@router.get("", response_model=CreditsResponse, summary="Get credit balance")
async def get_balance(current_user: dict = Depends(get_current_user)):
    """Get current user's credit balance"""
    data = await get_credits(current_user["id"])
    return data


@router.post("/earn", response_model=CreditsResponse, summary="Earn credits from ad")
async def earn_from_ad(request: EarnCreditsRequest, current_user: dict = Depends(get_current_user)):
    """Add credits after watching a rewarded video ad"""
    user_id = current_user["id"]

    # Anti-spam cooldown
    last_earn = await get_last_earn_time(user_id)
    if last_earn:
        last_dt = datetime.fromisoformat(last_earn)
        elapsed = (datetime.utcnow() - last_dt).total_seconds()
        if elapsed < EARN_COOLDOWN_SECONDS:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Please wait {int(EARN_COOLDOWN_SECONDS - elapsed)}s before watching another ad"
            )

    # Check if SSV already credited this reward (avoid double-credit)
    last_ssv = await get_last_ssv_time(user_id)
    if last_ssv:
        ssv_dt = datetime.fromisoformat(last_ssv)
        ssv_elapsed = (datetime.utcnow() - ssv_dt).total_seconds()
        if ssv_elapsed < 10:
            # SSV already credited within last 10s — just return current balance
            data = await get_credits(user_id)
            logger.info(f"User {user_id} earn skipped (SSV already credited {ssv_elapsed:.0f}s ago)")
            return {"balance": data["balance"], "last_updated": datetime.utcnow().isoformat()}

    new_balance = await add_credits(user_id, 3, "earn_ad")
    logger.info(f"User {user_id} earned 3 credits (ad: {request.ad_id}), balance: {new_balance}")

    return {"balance": new_balance, "last_updated": datetime.utcnow().isoformat()}


@router.post("/spend", response_model=SpendCreditsResponse, summary="Spend credits on prediction")
async def spend_on_prediction(request: SpendCreditsRequest, current_user: dict = Depends(get_current_user)):
    """Spend credits to view a crypto prediction"""
    user_id = current_user["id"]

    amount = request.amount if request.amount in (1, 3) else 3
    new_balance = await spend_credits(user_id, amount, request.crypto)
    if new_balance is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Insufficient credits. Watch an ad to earn more."
        )

    logger.info(f"User {user_id} spent {amount} credits on {request.crypto}, balance: {new_balance}")

    return {"success": True, "balance": new_balance, "crypto": request.crypto}


@router.get("/admob-callback", summary="AdMob SSV callback")
async def admob_ssv_callback(request: Request):
    """Google AdMob server-side verification callback.
    Google calls this URL after a user watches a rewarded ad.
    Query params include: ad_network, ad_unit, custom_data, key_id,
    reward_amount, reward_item, signature, timestamp, transaction_id, user_id
    """
    params = dict(request.query_params)

    user_id_str = params.get("user_id") or params.get("custom_data")
    transaction_id = params.get("transaction_id", "")
    reward_amount = int(params.get("reward_amount", 3))

    if not user_id_str:
        logger.warning(f"AdMob SSV: no user_id in callback params: {params}")
        raise HTTPException(status_code=400, detail="Missing user_id")

    try:
        user_id = int(user_id_str)
    except ValueError:
        logger.warning(f"AdMob SSV: invalid user_id={user_id_str}")
        raise HTTPException(status_code=400, detail="Invalid user_id")

    # Dedup: check if this transaction was already processed
    existing = await get_ssv_reward(transaction_id)
    if existing:
        logger.info(f"AdMob SSV: duplicate transaction_id={transaction_id}, skipping")
        return {"status": "already_processed"}

    # Credit the user and record transaction for dedup
    new_balance = await add_credits(user_id, reward_amount, "earn_ad_ssv")
    await record_ssv_reward(transaction_id, user_id, reward_amount)
    logger.info(f"AdMob SSV: user {user_id} earned {reward_amount} credits (tx={transaction_id}), balance={new_balance}")

    return {"status": "ok", "balance": new_balance}
