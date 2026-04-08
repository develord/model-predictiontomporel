# FIL (Filecoin) - Production Pipeline

## Overview
- **Pair**: FIL/USDT (Binance Futures)
- **Data start**: 2020-10-15 (FIL listing date)
- **Timeframes**: 4h, 1d, 1w (features) + 15min (backtest trade monitoring)
- **Architecture**: CNN LONG + CNN SHORT + XGBoost Meta-models
- **Template source**: Copied from SOL_PRODUCTION

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

## Data
- **1d candles**: 2,002 (Oct 2020 - Apr 2026)
- **Features**: 126 (66 per-TF + 11 cross-TF + 23 non-technical + 26 market regime)
- **Labeled samples**: 1,314 (TP=638, SL=676)
- **SHORT bear features**: +15 additional

---

## Model Architecture

### LONG Model: `CNNDirectionModel`
- **Conv layers**: kernel sizes 3, 5, 7 (parallel)
- **Temporal attention**: learns which timesteps matter
- **Params**: ~56K parameters
- **Training**: Seq=30, Batch=64, LR=0.0015, Epochs=200, Patience=35
- **Temperature**: 1.81
- **Validation WR**: 61-73% (best at >=75% conf: 73.1%)

### SHORT Model: `CNNDirectionModel` (same architecture)
- **Training**: Seq=30, Batch=32, LR=0.001, Epochs=250, Patience=60
- **Temperature**: 2.04
- **Validation WR**: 66.7% at >=50% conf

---

## Backtest Filters (V2 optimized)

### LONG Filters
| Filter | Condition | Purpose |
|--------|-----------|---------|
| Cooldown | 5 days after 2 consecutive losses | Avoid tilt |
| Momentum | At least 1/3 TFs bullish | Confirm trend |
| Weekly Momentum | `1w_momentum_5 >= -0.10` | Block LONG in weekly downtrend |
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

## TP/SL Configuration

### LONG (V2 optimized)
- **TP**: ATR-based, capped at 2.5%
- **SL**: TP * 0.65 (wider to survive normal swings)

### SHORT
- **TP**: 3% drop (fixed)
- **SL**: 3% rise (fixed)

---

## Backtest Results - Q1 2026 (83 days)

| Metric | Value |
|--------|-------|
| **Trades** | 10 (10 LONG + 0 SHORT) |
| **LONG WR** | **60%** (6 TP / 4 SL) |
| **Overall WR** | **60%** |
| **Return** | **+5.64%** |
| **Capital** | $1000 -> $1056 |

### Trade Details
| Date | Dir | Conf | Result | PnL |
|------|-----|------|--------|-----|
| Jan 3 | LONG | 89% | SL | -1.67% |
| Jan 4 | LONG | 88% | TP | +2.45% |
| Jan 5 | LONG | 76% | SL | -1.67% |
| Jan 6 | LONG | 68% | SL | -1.67% |
| Jan 12 | LONG | 92% | TP | +2.45% |
| Jan 13 | LONG | 94% | TP | +2.45% |
| Jan 14 | LONG | 93% | TP | +2.45% |
| Jan 15 | LONG | 94% | TP | +2.45% |
| Jan 16 | LONG | 94% | TP | +2.45% |
| Jan 17 | LONG | 95% | SL | -1.67% |

### Key Observations
- **LONG model dominates** on FIL - 6 consecutive TP trades from Jan 12-16
- **SHORT filtered** by bull_sma20 (5 times) - FIL was in uptrend in January
- **High conf = good signals**: all TP trades had conf >= 88%
- **Asymmetric TP/SL** (2.45% TP vs 1.67% SL) ensures profitability even at 60% WR

### Filtered Signals
- LONG: weak_momentum (28), bear_sma50 (20), cooldown (5), weak_weekly (3)
- SHORT: bull_sma20 (5), weak_bear_momentum (1)

---

## Coins Tested and Rejected
During the search for the 12th coin, we tested:
- **ATOM (Cosmos)**: -11.44%, LONG 0% WR, SHORT 45% WR - abandoned
- **POL (Polygon)**: -6.95%, SHORT overfit (91% validation -> 42% backtest) - abandoned
- **SUI**: +0.23%, SHORT 46% WR - breakeven, abandoned

FIL was the best performing new coin candidate.

---

## Live Trading Config

```python
'FIL': {
    'pair': 'FIL/USDT',
    'long_model': 'fil_cnn_model.pt',
    'short_model': 'fil_short_cnn_model.pt',
    'long_scaler': 'fil_feature_scaler.joblib',
    'short_scaler': 'fil_short_feature_scaler.joblib',
    'long_features': 'fil_features.json',
    'short_features': 'fil_short_features.json',
    'timeframes': ['4h', '1d', '1w'],
    'long_conf': 0.60,
    'short_conf': 0.55,
    'long_meta_conf': 0.0,
    'short_meta_conf': 0.0,
    'v3': True,
    'data_start': '2020-10-15',
}
```

---

## Files Deployed to LIVE_TRADING/models/
- `fil_cnn_model.pt` (LONG)
- `fil_short_cnn_model.pt` (SHORT)
- `fil_feature_scaler.joblib` / `fil_short_feature_scaler.joblib`
- `fil_features.json` / `fil_short_features.json`
