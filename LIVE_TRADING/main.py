"""
FastAPI Server - Crypto Predictions API
========================================
Serveur API pour les prédictions de cryptomonnaies avec modèles V11 TEMPORAL
"""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'training'))
sys.path.insert(0, str(project_root / 'data'))

from fastapi import FastAPI, HTTPException, Request, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime
import logging
import time as _time
from collections import defaultdict

from config import settings
from models import (
    PredictionResponse,
    AllPredictionsResponse,
    CryptoListResponse,
    HealthCheckResponse,
    ErrorResponse,
    BacktestRequest,
    BacktestResponse
)
from auth import verify_api_key, get_current_user
from auth_routes import router as auth_router
from credits_routes import router as credits_router
from notification_routes import router as notification_router
from database import init_db
from predictions_cnn import CNNPredictionService
try:
    from backtest_service import get_backtest_service
except ImportError:
    get_backtest_service = None

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app (Swagger disabled in production)
app = FastAPI(
    title="CryptoXHunter API",
    description="AI-Powered Crypto Trading Signals",
    version="3.0.0",
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
    openapi_url="/openapi.json" if settings.DEBUG else None,
)

# CORS configuration - restricted for mobile-only API
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include auth routes
app.include_router(auth_router)
app.include_router(credits_router)
app.include_router(notification_router)


# ============================================================================
# RATE LIMITING (in-memory, per-IP)
# ============================================================================
_rate_limits: dict = defaultdict(list)  # {ip: [timestamps]}
RATE_LIMIT_WINDOW = 60  # seconds
RATE_LIMITS = {
    '/auth/': 10,          # 10 auth attempts per minute
    '/api/credits/earn': 6,  # 6 earn requests per minute
    '/api/predictions/': 30, # 30 predictions per minute
    '_default': 60,          # 60 requests per minute for other endpoints
}


def _get_rate_limit(path: str) -> int:
    for prefix, limit in RATE_LIMITS.items():
        if prefix != '_default' and path.startswith(prefix):
            return limit
    return RATE_LIMITS['_default']


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Per-IP rate limiting"""
    client_ip = request.headers.get("X-Real-IP", request.client.host if request.client else "unknown")
    path = request.url.path
    limit = _get_rate_limit(path)
    now = _time.time()
    key = f"{client_ip}:{path.split('/')[1] if '/' in path[1:] else path}"

    # Clean old entries and check limit
    _rate_limits[key] = [t for t in _rate_limits[key] if now - t < RATE_LIMIT_WINDOW]
    if len(_rate_limits[key]) >= limit:
        return JSONResponse(
            status_code=429,
            content={"error": "Too many requests", "retry_after": RATE_LIMIT_WINDOW}
        )
    _rate_limits[key].append(now)

    return await call_next(request)


# API Key middleware - blocks requests without valid app API key
@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    """Verify X-API-Key header on all requests except health and public endpoints"""
    path = request.url.path
    exempt_exact = {"/health", "/"}
    exempt_prefixes = ("/api/analysis/", "/api/news", "/api/credits", "/auth/", "/api/notifications/signal-webhook", "/api/notifications/close-signal", "/api/notifications/history")

    if path in exempt_exact or path == "/api/analysis" or path == "/api/news" or any(path.startswith(p) for p in exempt_prefixes):
        return await call_next(request)

    api_key = request.headers.get("X-API-Key")
    if not api_key or not verify_api_key(api_key):
        return JSONResponse(
            status_code=403,
            content={"error": "Forbidden"}
        )

    return await call_next(request)


# Initialize services
prediction_service = None
backtest_service = None


@app.on_event("startup")
async def startup_event():
    """Load CNN models on startup"""
    global prediction_service, backtest_service
    try:
        # Initialize user database
        await init_db()

        logger.info("Starting API server with CNN+Meta V3 models...")
        prediction_service = CNNPredictionService()
        await prediction_service.load_models()
        logger.info(f"API ready - {len(prediction_service.models)} CNN models loaded")
        logger.info(f"Coins: BTC, ETH, SOL, DOGE, AVAX, XRP, LINK, ADA, NEAR (LONG + SHORT + Meta)")

        if get_backtest_service:
            backtest_service = get_backtest_service()
            logger.info("Backtest service initialized")

    except Exception as e:
        logger.error(f"Failed to load CNN models: {e}")
        prediction_service = CNNPredictionService()  # Empty service, will load on demand


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down API server...")


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get(
    "/",
    response_model=dict,
    summary="API Root",
    description="Point d'entrée de l'API avec informations de base"
)
async def root():
    """Welcome endpoint"""
    return {
        "message": "Crypto Predictions API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "cryptos": "/api/cryptos"
    }


@app.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="Health Check",
    description="Vérifier l'état du serveur et des modèles"
)
async def health_check():
    """Health check endpoint"""
    if prediction_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not initialized"
        )

    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": len(prediction_service.models),
        "cryptos_available": list(prediction_service.models.keys())
    }


@app.get(
    "/api/cryptos",
    response_model=CryptoListResponse,
    summary="Liste des Cryptos",
    description="Obtenir la liste de toutes les cryptomonnaies supportées"
)
async def get_cryptos(current_user: dict = Depends(get_current_user)):
    """Get list of supported cryptocurrencies"""
    if prediction_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not initialized"
        )

    # All supported cryptos with metadata
    cryptos_with_id = {
        "bitcoin": {
            "id": "bitcoin",
            "symbol": "BTCUSDT",
            "name": "Bitcoin",
            "models": ["CNN_LONG"],
            "status": "active"
        },
        "ethereum": {
            "id": "ethereum",
            "symbol": "ETHUSDT",
            "name": "Ethereum",
            "models": ["CNN_LONG", "CNN_SHORT"],
            "status": "active"
        },
        "solana": {
            "id": "solana",
            "symbol": "SOLUSDT",
            "name": "Solana",
            "models": ["CNN_LONG", "CNN_SHORT"],
            "status": "active"
        },
        "dogecoin": {
            "id": "dogecoin",
            "symbol": "DOGEUSDT",
            "name": "Dogecoin",
            "models": ["CNN_LONG", "CNN_SHORT"],
            "status": "active"
        },
        "avalanche": {
            "id": "avalanche",
            "symbol": "AVAXUSDT",
            "name": "Avalanche",
            "models": ["CNN_LONG", "CNN_SHORT"],
            "status": "active"
        },
        "xrp": {
            "id": "xrp",
            "symbol": "XRPUSDT",
            "name": "XRP",
            "models": ["CNN_LONG", "CNN_SHORT"],
            "status": "active"
        },
        "chainlink": {
            "id": "chainlink",
            "symbol": "LINKUSDT",
            "name": "Chainlink",
            "models": ["CNN_LONG", "CNN_SHORT"],
            "status": "active"
        },
        "cardano": {
            "id": "cardano",
            "symbol": "ADAUSDT",
            "name": "Cardano",
            "models": ["CNN_LONG", "CNN_SHORT"],
            "status": "active"
        },
        "near": {
            "id": "near",
            "symbol": "NEARUSDT",
            "name": "NEAR Protocol",
            "models": ["CNN_LONG", "CNN_SHORT"],
            "status": "active"
        },
        "polkadot": {
            "id": "polkadot",
            "symbol": "DOTUSDT",
            "name": "Polkadot",
            "models": ["CNN_LONG", "CNN_SHORT"],
            "status": "active"
        },
        "filecoin": {
            "id": "filecoin",
            "symbol": "FILUSDT",
            "name": "Filecoin",
            "models": ["CNN_LONG", "CNN_SHORT"],
            "status": "active"
        }
    }

    return {
        "cryptos": cryptos_with_id,
        "count": len(cryptos_with_id)
    }



@app.get(
    "/api/predictions/{crypto}",
    response_model=PredictionResponse,
    summary="Prédiction Crypto",
    description="Obtenir la prédiction pour une cryptomonnaie spécifique",
    responses={
        200: {
            "description": "Prédiction générée avec succès",
            "content": {
                "application/json": {
                    "example": {
                        "crypto": "bitcoin",
                        "symbol": "BTCUSDT",
                        "name": "Bitcoin",
                        "signal": "BUY",
                        "confidence": 0.65,
                        "probabilities": {
                            "buy": 0.65,
                            "sell": 0.15,
                            "hold": 0.20
                        },
                        "current_price": 45000.50,
                        "timestamp": "2025-01-01T12:00:00"
                    }
                }
            }
        },
        404: {
            "description": "Crypto non trouvée",
            "model": ErrorResponse
        }
    }
)
async def get_prediction(crypto: str, current_user: dict = Depends(get_current_user)):
    """Get prediction for specific cryptocurrency using CNN LONG+SHORT models"""
    if prediction_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not initialized"
        )

    crypto = crypto.lower()
    supported = ['bitcoin', 'ethereum', 'solana', 'dogecoin', 'avalanche', 'xrp', 'chainlink', 'cardano', 'near', 'polkadot', 'filecoin']
    if crypto not in supported:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Crypto '{crypto}' not found. Available: {supported}"
        )

    try:
        # Check prediction cache (4h)
        now = _time.time()
        cached = PREDICTION_CACHE.get(crypto)
        if cached and (now - cached["timestamp"]) < PREDICTION_CACHE_DURATION:
            logger.info(f"Cache hit for {crypto} prediction")
            return cached["data"]

        prediction = await prediction_service.predict_one(crypto)
        PREDICTION_CACHE[crypto] = {"data": prediction, "timestamp": now}
        return prediction
    except Exception as e:
        logger.error(f"Error predicting {crypto}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate prediction"
        )


@app.get(
    "/api/price/{crypto}",
    summary="Prix Actuel",
    description="Obtenir le prix actuel d'une cryptomonnaie (inclus dans la prédiction)"
)
async def get_current_price(crypto: str, current_user: dict = Depends(get_current_user)):
    """Get current price for cryptocurrency (from latest CSV data)"""
    if prediction_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not initialized"
        )

    crypto = crypto.lower()
    if crypto not in prediction_service.models:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Crypto '{crypto}' not found"
        )

    try:
        # Get price from latest features
        _, current_price = prediction_service.get_latest_features(crypto)

        symbols = {
            'bitcoin': 'BTCUSDT',
            'ethereum': 'ETHUSDT',
            'solana': 'SOLUSDT'
        }

        return {
            "crypto": crypto,
            "symbol": symbols[crypto],
            "price": round(current_price, 2),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting price for {crypto}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get price"
        )


@app.post(
    "/api/backtest",
    response_model=BacktestResponse,
    summary="Backtest Simulation",
    description="Exécuter un backtest sur une période personnalisée avec un crypto spécifique",
    responses={
        200: {
            "description": "Backtest exécuté avec succès",
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "crypto": "bitcoin",
                        "data": {
                            "metrics": {
                                "total_trades": 45,
                                "win_trades": 28,
                                "loss_trades": 15,
                                "open_trades": 2,
                                "win_rate": 0.622,
                                "total_roi": 22.56,
                                "avg_trade_roi": 0.501,
                                "sharpe_ratio": 1.85,
                                "max_drawdown": 5.2,
                                "avg_bars_held": 8.4,
                                "expected_value": 0.47,
                                "tp_pct": 1.5,
                                "sl_pct": 0.75,
                                "prob_threshold": 0.5
                            },
                            "trades": [],
                            "total_candles": 2184,
                            "start_date": "2024-01-01",
                            "end_date": "2024-12-31"
                        }
                    }
                }
            }
        },
        400: {
            "description": "Paramètres invalides",
            "model": ErrorResponse
        },
        404: {
            "description": "Données ou modèle non trouvés",
            "model": ErrorResponse
        }
    }
)
async def run_backtest(request: BacktestRequest, current_user: dict = Depends(get_current_user)):
    """
    Run backtest simulation on historical data

    Args:
        request: Backtest parameters (crypto, dates, TP/SL, threshold)

    Returns:
        Backtest results with trades and performance metrics
    """
    if backtest_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Backtest service not initialized"
        )

    # Validate crypto
    crypto = request.crypto.lower()
    valid_cryptos = ['bitcoin', 'ethereum', 'solana']
    if crypto not in valid_cryptos:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid crypto '{crypto}'. Valid options: {valid_cryptos}"
        )

    try:
        logger.info(f"Running backtest: {crypto} from {request.start_date} to {request.end_date}")

        # Run backtest
        results = backtest_service.run_backtest(
            crypto=crypto,
            start_date=request.start_date,
            end_date=request.end_date,
            tp_pct=request.tp_pct,
            sl_pct=request.sl_pct,
            prob_threshold=request.prob_threshold
        )

        return {
            "success": True,
            "crypto": crypto,
            "data": results
        }

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Required data not found"
        )
    except ValueError as e:
        logger.error(f"Invalid parameters: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid parameters"
        )
    except Exception as e:
        logger.error(f"Backtest error for {crypto}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Backtest failed"
        )


# ============================================================================
# TECHNICAL ANALYSIS ENDPOINT
# ============================================================================

@app.get("/api/analysis/{crypto}")
async def get_technical_analysis(crypto: str):
    """Get detailed technical analysis indicators for a crypto."""
    if prediction_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    crypto = crypto.lower()
    try:
        # Check analysis cache (4h)
        now = _time.time()
        cached = ANALYSIS_CACHE.get(crypto)
        if cached and (now - cached["timestamp"]) < ANALYSIS_CACHE_DURATION:
            return cached["data"]

        result = await prediction_service.get_technical_analysis(crypto)
        ANALYSIS_CACHE[crypto] = {"data": result, "timestamp": now}
        return result
    except ValueError:
        raise HTTPException(status_code=404, detail="Crypto not found")
    except Exception as e:
        logger.error(f"Analysis error for {crypto}: {e}")
        raise HTTPException(status_code=500, detail="Analysis failed")


# ============================================================================
# NEWS ENDPOINT — Fetch + classify crypto news
# ============================================================================

import aiohttp
import re
import hashlib

NEWS_CACHE = {"data": None, "timestamp": 0}
NEWS_CACHE_DURATION = 21600  # 6 hours

# Prediction cache: 4 hours per crypto (models retrain daily, predictions stable intraday)
PREDICTION_CACHE: dict = {}  # {crypto: {"data": ..., "timestamp": float}}
PREDICTION_CACHE_DURATION = 14400  # 4 hours

# Analysis cache: 4 hours
ANALYSIS_CACHE: dict = {}
ANALYSIS_CACHE_DURATION = 3600  # 1 hour

# Keyword-based sentiment classifier (fast, no ML model needed on server)
BULLISH_KEYWORDS = [
    'rally', 'surge', 'soar', 'bull', 'breakout', 'all-time high', 'ath',
    'adoption', 'approve', 'approval', 'etf', 'institutional', 'partnership',
    'upgrade', 'launch', 'record', 'gain', 'pump', 'moon', 'recover',
    'bullish', 'accumulate', 'buy', 'growth', 'positive', 'boost',
    'milestone', 'integration', 'support', 'optimistic', 'profit',
]
BEARISH_KEYWORDS = [
    'crash', 'plunge', 'dump', 'bear', 'sell-off', 'selloff', 'hack',
    'exploit', 'ban', 'regulation', 'sec', 'lawsuit', 'fraud', 'scam',
    'liquidat', 'bankrupt', 'collapse', 'fear', 'panic', 'decline',
    'bearish', 'risk', 'warning', 'loss', 'drop', 'fall', 'negative',
    'investigation', 'crackdown', 'vulnerability', 'attack',
]

COIN_TAGS = {
    'bitcoin': ['bitcoin', 'btc'],
    'ethereum': ['ethereum', 'eth', 'vitalik'],
    'solana': ['solana', 'sol'],
    'xrp': ['xrp', 'ripple'],
    'cardano': ['cardano', 'ada'],
    'chainlink': ['chainlink', 'link'],
    'near': ['near protocol', 'near'],
    'avalanche': ['avalanche', 'avax'],
    'dogecoin': ['dogecoin', 'doge'],
}


def classify_sentiment(title: str, body: str = '') -> dict:
    """Rule-based sentiment classification."""
    text = f"{title} {body}".lower()
    bull_score = sum(1 for kw in BULLISH_KEYWORDS if kw in text)
    bear_score = sum(1 for kw in BEARISH_KEYWORDS if kw in text)
    total = bull_score + bear_score
    if total == 0:
        return {'label': 'neutral', 'score': 0.5}
    ratio = bull_score / total
    if ratio >= 0.65:
        return {'label': 'bullish', 'score': round(ratio, 2)}
    elif ratio <= 0.35:
        return {'label': 'bearish', 'score': round(1 - ratio, 2)}
    return {'label': 'neutral', 'score': 0.5}


def tag_coins(title: str, body: str = '') -> list:
    """Tag which coins a news article relates to."""
    text = f"{title} {body}".lower()
    tags = []
    for coin, keywords in COIN_TAGS.items():
        if any(kw in text for kw in keywords):
            tags.append(coin)
    return tags if tags else ['market']


@app.get("/api/news")
async def get_crypto_news():
    """Fetch and classify latest crypto news."""
    global NEWS_CACHE
    now = datetime.now().timestamp()

    # Return cached if fresh
    if NEWS_CACHE["data"] and (now - NEWS_CACHE["timestamp"]) < NEWS_CACHE_DURATION:
        return NEWS_CACHE["data"]

    articles = []

    # Source 1: CoinGecko News API (free, no key)
    try:
        async with aiohttp.ClientSession() as session:
            url = "https://api.coingecko.com/api/v3/news?page=1"
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    for item in data.get('data', [])[:25]:
                        title = item.get('title', '')
                        desc = item.get('description', '') or ''
                        sentiment = classify_sentiment(title, desc)
                        coins = tag_coins(title, desc)
                        article_id = hashlib.md5(title.encode()).hexdigest()[:12]
                        thumb = item.get('thumb_2x', '') or item.get('thumb', '') or ''
                        articles.append({
                            'id': article_id,
                            'title': title,
                            'source': item.get('author', 'Unknown'),
                            'url': item.get('url', ''),
                            'published_at': item.get('updated_at', '') or item.get('created_at', ''),
                            'sentiment': sentiment,
                            'coins': coins,
                            'image': thumb,
                        })
                    logger.info(f"CoinGecko news: {len(articles)} articles")
    except Exception as e:
        logger.warning(f"CoinGecko news failed: {e}")

    # Source 2: CoinTelegraph RSS (fallback)
    if not articles:
        try:
            try:
                import defusedxml.ElementTree as ET
            except ImportError:
                import xml.etree.ElementTree as ET
            async with aiohttp.ClientSession() as session:
                url = "https://cointelegraph.com/rss"
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    if resp.status == 200:
                        text = await resp.text()
                        root = ET.fromstring(text)
                        for item in root.findall('.//item')[:25]:
                            title = item.findtext('title', '')
                            desc_raw = item.findtext('description', '') or ''
                            # Strip HTML from description
                            desc = re.sub(r'<[^>]+>', '', desc_raw).strip()
                            link = item.findtext('link', '')
                            # Handle CDATA in link
                            if link:
                                link = link.strip().split('?')[0]
                            pub_date = item.findtext('pubDate', '')
                            creator = item.findtext('{http://purl.org/dc/elements/1.1/}creator', 'CoinTelegraph')
                            sentiment = classify_sentiment(title, desc)
                            coins = tag_coins(title, desc)
                            article_id = hashlib.md5(title.encode()).hexdigest()[:12]
                            # Get image from media:content
                            media = item.find('{http://search.yahoo.com/mrss/}content')
                            image = media.get('url', '') if media is not None else ''
                            articles.append({
                                'id': article_id,
                                'title': title,
                                'source': creator,
                                'url': link,
                                'published_at': pub_date,
                                'sentiment': sentiment,
                                'coins': coins,
                                'image': image,
                            })
                        logger.info(f"CoinTelegraph RSS: {len(articles)} articles")
        except Exception as e:
            logger.warning(f"CoinTelegraph RSS failed: {e}")

    result = {
        'articles': articles,
        'count': len(articles),
        'timestamp': datetime.now().isoformat(),
    }

    NEWS_CACHE = {"data": result, "timestamp": now}
    return result


# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler — never leak internal details"""
    logger.error(f"Unhandled exception on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )
