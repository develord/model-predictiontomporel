"""
Credits Routes
===============
Earn credits (ad rewards) + spend credits (view predictions)
"""
from fastapi import APIRouter, HTTPException, Depends, status
from datetime import datetime
import logging

from auth import get_current_user
from database import get_credits, add_credits, spend_credits, get_last_earn_time
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

    new_balance = await add_credits(user_id, 3, "earn_ad")
    logger.info(f"User {user_id} earned 3 credits (ad: {request.ad_id}), balance: {new_balance}")

    return {"balance": new_balance, "last_updated": datetime.utcnow().isoformat()}


@router.post("/spend", response_model=SpendCreditsResponse, summary="Spend credits on prediction")
async def spend_on_prediction(request: SpendCreditsRequest, current_user: dict = Depends(get_current_user)):
    """Spend credits to view a crypto prediction"""
    user_id = current_user["id"]

    new_balance = await spend_credits(user_id, 3, request.crypto)
    if new_balance is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Insufficient credits. Watch an ad to earn more."
        )

    logger.info(f"User {user_id} spent 3 credits on {request.crypto}, balance: {new_balance}")

    return {"success": True, "balance": new_balance, "crypto": request.crypto}
