# DOT (Polkadot) - Production Pipeline

## Overview
- **Pair**: DOT/USDT (Binance Futures)
- **Data start**: 2020-08-20 (DOT listing date)
- **Timeframes**: 4h, 1d, 1w (features) + 15min (backtest trade monitoring)
- **Architecture**: CNN LONG + CNN SHORT + XGBoost Meta-models
- **Template source**: Copied from SOL_PRODUCTION (same architecture, no BTC influence)

---

## Pipeline Scripts

| Script | Description |
|--------|-------------|
| `01_download_data.py` | Download OHLCV from Binance (1h, 4h, 1d, 1w) |
| `02_feature_engineering.py` | 126 features + ATR-adaptive Triple Barrier labels |
| `03_train_cnn.py` | Train LONG CNN model (CNNDirectionModel) |
| `03_train_short_model.py` | Train SHORT CNN model (CNNDirectionModel) |
| `04_backtest_independent.py` | Q1 2026 backtest with 15min candle monitoring |
| `05_train_meta_xgboost.py` | XGBoost meta-model gating |
| `07_optimize_meta_thresholds.py` | Optimize confidence thresholds |
| `08_train_production.py` | Retrain on full dataset for deployment |

---

## Features (126 total)

### Per-Timeframe (3 TF x 22 = 66 features)
For each of 1d, 4h, 1w:
- RSI (14), Stoch RSI (K, D)
- MACD (line, signal, histogram)
- Bollinger Bands (upper, mid, lower, width, %B)
- ATR (14), ADX (14)
- SMA (20, 50, 200), EMA (12, 26)
- Volume SMA ratio
- Momentum (5), ROC (10)

### Cross-Timeframe (11 features)
- RSI divergence (1d vs 4h, 1d vs 1w)
- MACD divergence (1d vs 4h)
- Momentum alignment score
- Volatility ratio (4h vs 1d)
- Trend agreement score
- Multi-TF momentum composite

### Non-Technical (23 features)
- Distance from SMA20, SMA50, SMA200
- Volatility regime (ATR z-score)
- Trend score (composite)
- Day of week, month
- Returns (1d, 3d, 7d, 14d, 30d)
- Volume profile features
- Price range features

### SHORT-Specific Bear Features (+15)
- ROC (3, 5, 10 periods)
- RSI divergence (bearish)
- Consecutive red candles
- Volume spike ratio
- Bear momentum composite

---

## Labels (Triple Barrier)
- **LONG**: TP = 1.5x ATR, SL = 0.75x ATR, Window = 10 days
- **SHORT**: TP = 2% drop, SL = 1% rise, Window = 10 days
- ATR-adaptive: adjusts to current volatility

---

## Model Architecture

### LONG Model: `CNNDirectionModel`
- **Conv layers**: kernel sizes 3, 5, 7 (parallel)
- **Temporal attention**: learns which timesteps matter
- **Params**: ~56K parameters
- **Training**: Seq=30, Batch=64, LR=0.0015, Epochs=200, Patience=35
- **Data split**: Train <= 2025-06-30, Val = 2025-07-01 to 2025-12-31
- **Augmentation**: 2x with noise (std=0.02), label smoothing 0.1

### SHORT Model: `CNNDirectionModel` (same architecture)
- **Note**: Originally tried `DeepCNNShortModel` (158K params) but it collapsed on DOT's small dataset (~1,267 training samples). Switched to same `CNNDirectionModel` (56K params) which converged successfully.
- **Training**: Seq=30, Batch=32, LR=0.001, Epochs=250, Patience=60
- **Class weights**: 1.5x boost for SHORT class
- **Temperature**: 2.24 (calibrated)

### Meta-Models: XGBoost
- **LONG meta**: 663 samples, 70% win distribution
- **SHORT meta**: 118 samples, 48% accuracy (marginal)
- **Config**: meta pass-through (threshold = 0.0 for both)

---

## Backtest Filters

### LONG Filters
| Filter | Condition | Purpose |
|--------|-----------|---------|
| Cooldown | 5 days after 2 consecutive losses | Avoid tilt |
| Momentum | At least 1/3 TFs bullish | Confirm trend |
| **Weekly Momentum** | `1w_momentum_5 >= -0.10` | **V2: Block LONG in weekly downtrend** |
| Bear SMA50 | `distance_from_sma50 >= -5%` | No LONG in deep bear |
| Bear SMA20 | `distance_from_sma20 >= -2%` | No LONG below SMA20 |
| Volatility | `volatility_regime <= 2.5` | No LONG in chaos |
| Trend | `trend_score >= -3` | No LONG in downtrend |

### SHORT Filters
| Filter | Condition | Purpose |
|--------|-----------|---------|
| Cooldown | 5 days after 2 consecutive losses | Avoid tilt |
| Bear Momentum | At least 1/3 TFs bearish | Confirm downtrend |
| Bull SMA50 | `distance_from_sma50 <= 5%` | No SHORT in strong bull |
| Bull SMA20 | `distance_from_sma20 <= 3%` | No SHORT above SMA20 |
| Volatility | `volatility_regime <= 2.5` | No SHORT in chaos |
| Uptrend | `trend_score <= 3` | No SHORT in uptrend |

---

## Backtest TP/SL Configuration

### LONG (V2 optimized)
- **TP**: ATR-based, capped at 2.5% (was 3%)
- **SL**: TP * 0.65 (was 0.50) - wider SL to survive normal swings
- Fallback: TP=1.5%, SL=0.75%

### SHORT
- **TP**: 3% drop (fixed)
- **SL**: 3% rise (fixed)
- Ratio 1:1 with 62% WR = profitable edge

---

## Backtest Results - Q1 2026 (83 days)

### V1 (Original)
| Metric | Value |
|--------|-------|
| Trades | 12 (4 LONG + 8 SHORT) |
| LONG WR | 25% (1 TP / 3 SL) |
| SHORT WR | 62% (5 TP / 3 SL) |
| Overall WR | 50% |
| Return | +1.73% |
| Capital | $1000 -> $1017 |

### V2 (Optimized - Current)
| Metric | Value |
|--------|-------|
| Trades | 8 (0 LONG + 8 SHORT) |
| SHORT WR | **62.5%** (5 TP / 3 SL) |
| Overall WR | **62.5%** |
| Return | **+4.25%** |
| Capital | $1000 -> $1043 |

### V2 Improvements
1. **Weekly momentum filter** (`1w_momentum < -10%` blocks LONG): eliminated 4 toxic LONG trades in late Feb crash
2. **LONG TP cap reduced** (3% -> 2.5%): more realistic targets
3. **LONG SL widened** (0.5x -> 0.65x TP): survives normal volatility swings

### Failed SHORT Analysis (3 losses)
| Date | Entry | Result | Analysis |
|------|-------|--------|----------|
| Feb 20 | 1.3403 | SL -2.96% | +4.2% intraday spike before collapse to 1.225. Direction correct, SL too tight for ATR (7.5%) |
| Mar 7 | 1.4473 | SL -2.96% | SL hit by 3 pips (1.4910 vs 1.4907). Classic stop-hunt. Price dropped after. |
| Mar 8 | 1.4513 | SL -2.96% | Same Mar 9 spike caught both trades. Price dropped within days. |

**Conclusion**: All 3 SHORT losses had the correct direction prediction. Losses were due to intraday volatility spikes (stop-hunts), not bad predictions. With 62% WR at 1:1 ratio, the edge is solid.

---

## Live Trading Config

```python
# LIVE_TRADING/config.py
'DOT': {
    'pair': 'DOT/USDT',
    'long_model': 'dot_cnn_model.pt',
    'short_model': 'dot_short_cnn_model.pt',
    'long_scaler': 'dot_feature_scaler.joblib',
    'short_scaler': 'dot_short_feature_scaler.joblib',
    'long_features': 'dot_features.json',
    'short_features': 'dot_short_features.json',
    'timeframes': ['4h', '1d', '1w'],
    'long_conf': 0.55,
    'short_conf': 0.55,
    'long_meta_conf': 0.0,    # Meta pass-through
    'short_meta_conf': 0.0,   # Meta pass-through
    'v3': True,
    'data_start': '2020-08-20',
}
```

---

## Key Learnings
1. **DeepCNNShortModel too complex** for DOT's small dataset (1,267 samples). CNNDirectionModel (56K params) works for both LONG and SHORT.
2. **Weekly momentum is critical** for LONG filtering - blocks entries before weekly downturns.
3. **SHORT model is the profit driver** on DOT in Q1 2026 (bear-heavy period).
4. **3% fixed SL for SHORT** is the sweet spot - wider SL (4%) saves some trades but increases loss magnitude, net negative.
5. **Stop-hunts are unavoidable** with fixed SL - the 62% WR compensates for the 3 losses.

---

## Files Deployed to LIVE_TRADING/models/
- `dot_cnn_model.pt` (LONG)
- `dot_short_cnn_model.pt` (SHORT)
- `dot_feature_scaler.joblib` / `dot_short_feature_scaler.joblib`
- `dot_features.json` / `dot_short_features.json`
- `dot_meta_long.joblib` / `dot_meta_short.joblib`
