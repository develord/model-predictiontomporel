"""
Pydantic Models for API
=======================
"""
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from datetime import datetime


class Probabilities(BaseModel):
    """Prediction probabilities"""
    buy: float = Field(..., description="Probabilité BUY (0-1)", ge=0, le=1)
    sell: float = Field(..., description="Probabilité SELL (0-1)", ge=0, le=1)
    hold: float = Field(..., description="Probabilité HOLD (0-1)", ge=0, le=1)


class RiskManagement(BaseModel):
    """Risk management metrics"""
    target_price: Optional[float] = Field(None, description="Prix cible")
    stop_loss: Optional[float] = Field(None, description="Prix stop loss")
    take_profit: Optional[float] = Field(None, description="Prix take profit")
    take_profit_pct: Optional[float] = Field(None, description="TP en %")
    stop_loss_pct: Optional[float] = Field(None, description="SL en %")
    risk_reward_ratio: Optional[float] = Field(None, description="Ratio Risk:Reward")
    potential_gain_percent: Optional[float] = Field(None, description="Gain potentiel en %")
    potential_loss_percent: Optional[float] = Field(None, description="Perte potentielle en %")


class PredictionResponse(BaseModel):
    """Response model for single crypto prediction"""
    crypto: str = Field(..., description="Crypto ID (bitcoin, ethereum, etc.)")
    symbol: str = Field(..., description="Trading symbol (BTCUSDT, ETHUSDT, etc.)")
    name: str = Field(..., description="Nom complet (Bitcoin, Ethereum, etc.)")
    signal: str = Field(..., description="Signal de trading: BUY, SELL, ou HOLD")
    direction: Optional[str] = Field(None, description="Direction: LONG, SHORT, ou null")
    confidence: float = Field(..., description="Confiance du signal actif", ge=0, le=1)
    long_confidence: Optional[float] = Field(None, description="Confiance LONG (0-1)")
    short_confidence: Optional[float] = Field(None, description="Confiance SHORT (0-1)")
    long_filter: Optional[str] = Field(None, description="Raison filtre LONG (bear_market, etc.)")
    short_filter: Optional[str] = Field(None, description="Raison filtre SHORT (bull_market, etc.)")
    meta_long_prob: Optional[float] = Field(None, description="Meta-model LONG probability")
    meta_short_prob: Optional[float] = Field(None, description="Meta-model SHORT probability")
    probabilities: Optional[Probabilities] = Field(None, description="Probabilités détaillées (legacy)")
    threshold: Optional[float] = Field(None, description="Threshold optimal", ge=0, le=1)
    current_price: Optional[float] = Field(None, description="Prix actuel en USDT")
    risk_management: Optional[RiskManagement] = Field(None, description="Gestion de risque (TP, SL, R:R)")
    model: Optional[str] = Field(None, description="Nom du modèle (CNN_1D_MultiScale + XGBoost_Meta)")
    features: Optional[str] = Field(None, description="Features utilisées")
    data_source: Optional[str] = Field(None, description="Source des données (binance_live)")
    model_version: Optional[str] = Field(None, description="Version du modèle")
    timestamp: str = Field(..., description="Timestamp ISO de la prédiction")

    model_config = {
        "json_schema_extra": {
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
                "risk_management": {
                    "target_price": 49500.0,
                    "stop_loss": 44100.0,
                    "take_profit": 49500.0,
                    "risk_reward_ratio": 5.0,
                    "potential_gain_percent": 10.0,
                    "potential_loss_percent": 2.0
                },
                "timestamp": "2025-01-01T12:00:00"
            }
        }
    }


class AllPredictionsResponse(BaseModel):
    """Response model for all predictions"""
    predictions: Dict[str, PredictionResponse] = Field(..., description="Prédictions par crypto")
    timestamp: str = Field(..., description="Timestamp ISO")
    count: int = Field(..., description="Nombre de cryptos")


class CryptoInfo(BaseModel):
    """Crypto information"""
    id: str = Field(..., description="Crypto ID")
    symbol: str = Field(..., description="Trading symbol")
    name: str = Field(..., description="Nom complet")
    models: Optional[List[str]] = Field(None, description="Modèles disponibles")
    status: Optional[str] = Field(None, description="Statut")


class CryptoListResponse(BaseModel):
    """Response model for crypto list"""
    cryptos: Dict[str, CryptoInfo] = Field(..., description="Liste des cryptos supportées")
    count: int = Field(..., description="Nombre de cryptos")


class HealthCheckResponse(BaseModel):
    """Response model for health check"""
    status: str = Field(..., description="État du service (healthy/unhealthy)")
    timestamp: str = Field(..., description="Timestamp ISO")
    models_loaded: int = Field(..., description="Nombre de modèles chargés")
    cryptos_available: List[str] = Field(..., description="Liste des cryptos disponibles")


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Message d'erreur")
    detail: Optional[str] = Field(None, description="Détails supplémentaires")
    timestamp: str = Field(..., description="Timestamp ISO")


# ============================================================================
# BACKTEST MODELS
# ============================================================================

# ============================================================================
# AUTH MODELS
# ============================================================================

class GoogleAuthRequest(BaseModel):
    """Google Sign-In authentication request"""
    id_token: str = Field(..., description="Google ID token from the app")


class BinanceAuthRequest(BaseModel):
    """Binance OAuth2 authentication request"""
    code: str = Field(..., description="Authorization code from Binance OAuth")
    redirect_uri: str = Field(..., description="Redirect URI used in the OAuth flow")


class RefreshTokenRequest(BaseModel):
    """Token refresh request"""
    refresh_token: str = Field(..., description="Refresh token to exchange for a new access token")


class UserResponse(BaseModel):
    """User information response"""
    id: int = Field(..., description="User ID")
    email: Optional[str] = Field(None, description="User email")
    name: Optional[str] = Field(None, description="User display name")
    avatar: Optional[str] = Field(None, description="User avatar URL")
    auth_provider: str = Field(..., description="Authentication provider (google/binance)")
    created_at: str = Field(..., description="Account creation timestamp")
    last_login: str = Field(..., description="Last login timestamp")


class AuthResponse(BaseModel):
    """Authentication response with tokens"""
    access_token: str = Field(..., description="JWT access token")
    refresh_token: str = Field(..., description="JWT refresh token for renewal")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiry in seconds")
    user: UserResponse = Field(..., description="Authenticated user info")


# ============================================================================
# CREDITS MODELS
# ============================================================================

class CreditsResponse(BaseModel):
    """Credit balance response"""
    balance: int = Field(..., description="Current credit balance")
    last_updated: str = Field(..., description="Last update timestamp")


class EarnCreditsRequest(BaseModel):
    """Request to earn credits after watching ad"""
    ad_id: str = Field(..., description="Ad unit ID that was watched")
    reward_amount: int = Field(default=3, description="Credits to add")


class SpendCreditsRequest(BaseModel):
    """Request to spend credits on a prediction"""
    crypto: str = Field(..., description="Crypto being viewed")
    amount: int = Field(default=3, description="Credits to spend")


class SpendCreditsResponse(BaseModel):
    """Response after spending credits"""
    success: bool = Field(..., description="Whether spend was successful")
    balance: int = Field(..., description="New balance after spend")
    crypto: str = Field(..., description="Crypto that was unlocked")


# ============================================================================
# BACKTEST MODELS
# ============================================================================

class BacktestRequest(BaseModel):
    """Request model for backtest"""
    crypto: str = Field(..., description="Crypto (bitcoin, ethereum, solana)")
    start_date: str = Field(..., description="Date de début (YYYY-MM-DD)")
    end_date: str = Field(..., description="Date de fin (YYYY-MM-DD)")
    tp_pct: Optional[float] = Field(1.5, description="Take profit %", gt=0)
    sl_pct: Optional[float] = Field(0.75, description="Stop loss %", gt=0)
    prob_threshold: Optional[float] = Field(0.5, description="Seuil de probabilité", ge=0, le=1)

    model_config = {
        "json_schema_extra": {
            "example": {
                "crypto": "bitcoin",
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "tp_pct": 1.5,
                "sl_pct": 0.75,
                "prob_threshold": 0.5
            }
        }
    }


class BacktestTrade(BaseModel):
    """Individual trade from backtest"""
    entry_date: str = Field(..., description="Date d'entrée")
    exit_date: str = Field(..., description="Date de sortie")
    entry_price: float = Field(..., description="Prix d'entrée", gt=0)
    exit_price: float = Field(..., description="Prix de sortie", gt=0)
    pnl_pct: float = Field(..., description="P&L en %")
    pnl_usd: float = Field(..., description="P&L en USD")
    outcome: str = Field(..., description="Résultat: WIN, LOSS, OPEN")
    duration_hours: int = Field(..., description="Durée en heures", ge=0)


class BacktestMetrics(BaseModel):
    """Performance metrics from backtest"""
    total_trades: int = Field(..., description="Nombre total de trades", ge=0)
    win_trades: int = Field(..., description="Trades gagnants", ge=0)
    loss_trades: int = Field(..., description="Trades perdants", ge=0)
    open_trades: int = Field(..., description="Trades ouverts", ge=0)
    win_rate: float = Field(..., description="Taux de réussite", ge=0, le=1)
    total_roi: float = Field(..., description="ROI total en %")
    avg_trade_roi: float = Field(..., description="ROI moyen par trade en %")
    sharpe_ratio: float = Field(..., description="Ratio de Sharpe")
    max_drawdown: float = Field(..., description="Drawdown maximum en %", ge=0)
    avg_bars_held: float = Field(..., description="Durée moyenne de détention (barres)", ge=0)
    expected_value: float = Field(..., description="Valeur attendue (EV) en %")
    tp_pct: float = Field(..., description="Take profit % utilisé", gt=0)
    sl_pct: float = Field(..., description="Stop loss % utilisé", gt=0)
    prob_threshold: float = Field(..., description="Seuil de probabilité utilisé", ge=0, le=1)


class BacktestData(BaseModel):
    """Backtest result data"""
    metrics: BacktestMetrics = Field(..., description="Métriques de performance")
    trades: List[BacktestTrade] = Field(..., description="Liste des trades")
    total_candles: int = Field(..., description="Nombre total de bougies", ge=0)
    start_date: str = Field(..., description="Date de début")
    end_date: str = Field(..., description="Date de fin")


class BacktestResponse(BaseModel):
    """Response model for backtest"""
    success: bool = Field(..., description="Succès de l'opération")
    crypto: str = Field(..., description="Crypto testée")
    data: BacktestData = Field(..., description="Résultats du backtest")

    model_config = {
        "json_schema_extra": {
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
                    "trades": [
                        {
                            "entry_date": "2024-01-15 08:00:00",
                            "exit_date": "2024-01-16 12:00:00",
                            "entry_price": 42500.0,
                            "exit_price": 43137.5,
                            "pnl_pct": 1.5,
                            "pnl_usd": 637.5,
                            "outcome": "WIN",
                            "duration_hours": 28
                        }
                    ],
                    "total_candles": 2184,
                    "start_date": "2024-01-01",
                    "end_date": "2024-12-31"
                }
            }
        }
    }
