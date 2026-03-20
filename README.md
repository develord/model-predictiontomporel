# 🚀 Crypto V10 - Multi-Timeframe Trading Model

## 📊 Overview

Advanced cryptocurrency trading model using multi-timeframe analysis (4h, 1d, 1w) with XGBoost ensemble.

**Cryptos**: Bitcoin (BTC), Ethereum (ETH), Solana (SOL)
**Timeframes**: 4h, 1d, 1w
**Model**: XGBoost Ensemble with multi-TF features

---

## 🎯 Key Improvements from V9

### ✅ What Worked in V9:
- **BTC Influence Features (37)**: ETH achieved +70.51% ROI with BTC correlation
- **Temporal Features (49)**: Reduced overfitting by 7%
- **Confidence Filtering**: Critical for performance (50% for ETH, 40% for BTC)
- **Optimized TP/SL**: Per-crypto optimization crucial

### 🚀 New in V10:
1. **Multi-Timeframe Analysis**:
   - 4h: Capture short-term momentum and entry/exit signals
   - 1d: Medium-term trends and swing trading (V9's timeframe)
   - 1w: Long-term trend confirmation and market regime

2. **Ensemble Model**:
   - Individual models per timeframe
   - Weighted voting based on timeframe performance
   - Confidence aggregation across timeframes

3. **Advanced Features**:
   - Cross-timeframe momentum alignment
   - Multi-TF support/resistance levels
   - Volume profile across timeframes
   - Market regime detection (trending vs ranging)

4. **Improved Risk Management**:
   - Dynamic TP/SL based on timeframe and volatility
   - Multi-TF signal confirmation
   - Drawdown protection

---

## 📁 Project Structure

```
crypto_v10_multi_tf/
├── README.md                   # This file
├── config/
│   ├── cryptos.json           # BTC, ETH, SOL configuration
│   └── timeframes.json        # 4h, 1d, 1w settings
├── data/
│   ├── data_manager_multi_tf.py   # Multi-TF data fetching
│   └── cache/                     # Cached data by TF
├── features/
│   ├── base_indicators.py         # RSI, MACD, Bollinger, etc.
│   ├── temporal_features.py       # Lag, momentum, patterns (V8)
│   ├── btc_influence.py           # BTC correlation features (V9)
│   ├── multi_tf_features.py       # NEW: Cross-TF features
│   └── volume_analysis.py         # NEW: Advanced volume features
├── models/
│   ├── ensemble_model.py          # Multi-TF ensemble
│   └── saved/                     # Trained models
├── training/
│   ├── train_multi_tf.py          # Training pipeline
│   └── optuna_optimize.py         # Hyperparameter tuning
├── backtesting/
│   ├── backtest_multi_tf.py       # Multi-TF backtest
│   └── performance_analysis.py    # Detailed analysis
├── production/
│   └── predict_realtime.py        # Real-time predictions
└── results/                       # Backtest results & reports
```

---

## 🔬 V9 Performance Summary

**Best Configuration (V9 Optimized)**:
| Crypto   | ROI      | Trades | Win Rate | Confidence | TP/SL   |
|----------|----------|--------|----------|------------|---------|
| Bitcoin  | +8.82%   | 7      | 57.14%   | 40%        | 5.0/3.0 |
| Ethereum | +70.51%  | 4      | 50.00%   | 50%        | 5.0/3.0 |
| Solana   | -18.52%  | 4      | 0.00%    | 55%        | 7.0/3.5 |
| **Total**| **+20.27%** | 15 | **46.67%** | -       | -       |

**Without Confidence Filter**: -13.58% ROI (catastrophic)
**Conclusion**: Confidence filtering is ESSENTIAL

---

## 🎓 Lessons Learned

### Critical Success Factors:
1. **Signal Quality > Signal Quantity**: 4 high-confidence trades (70% ROI) > 74 low-confidence (-16% ROI)
2. **BTC Influence Matters**: Altcoins heavily influenced by BTC movements
3. **Temporal Patterns**: Lag and momentum features reduce overfitting
4. **Per-Crypto Optimization**: BTC != ETH != SOL in behavior

### What to Avoid:
1. Taking all signals without confidence filtering
2. Using single timeframe (miss short/long term signals)
3. Uniform TP/SL across all cryptos
4. Ignoring BTC correlation for altcoins

---

## 🚀 Getting Started

### 1. Data Collection
```bash
python data/data_manager_multi_tf.py --crypto all --timeframes 4h,1d,1w
```

### 2. Feature Engineering
```bash
python features/generate_all_features.py --crypto BTC --timeframe 1d
```

### 3. Training
```bash
python training/train_multi_tf.py --crypto ETH --optimize
```

### 4. Backtesting
```bash
python backtesting/backtest_multi_tf.py --start 2025-01-01 --end 2026-03-20
```

### 5. Real-time Prediction
```bash
python production/predict_realtime.py --crypto all
```

---

## 📈 Expected Improvements in V10

Based on V9 analysis and multi-timeframe approach:

**Target ROI**: 50-70% (portfolio)
- Bitcoin: 15-25% (improved from 8.82%)
- Ethereum: 80-100% (improved from 70.51%)
- Solana: 10-20% (improved from -18.52%)

**Key Metrics**:
- Win Rate: 55-65% (from 46.67%)
- Trades: 20-40 per crypto
- Sharpe Ratio: >2.0
- Max Drawdown: <15%

---

## 🔧 Configuration

See `config/` for detailed settings:
- `cryptos.json`: Crypto-specific params (TP/SL, confidence thresholds)
- `timeframes.json`: Timeframe weights and lookahead periods

---

## 📊 Feature Summary

**Total Features per Crypto**:
- BTC: ~120 features (base + temporal + multi-TF)
- ETH/SOL: ~160 features (+ BTC influence + cross-crypto)

**Feature Categories**:
1. **Base Indicators (30)**: RSI, MACD, Bollinger, ATR, etc.
2. **Temporal (49)**: Lag, momentum, acceleration, patterns
3. **BTC Influence (37)**: Correlation, divergence, dominance
4. **Multi-TF (25)**: Cross-TF alignment, regime detection
5. **Volume (20)**: Volume profile, OBV, CMF, accumulation

---

## 🎯 Next Steps

1. Implement data manager multi-TF
2. Create multi-TF feature engineering
3. Build ensemble model architecture
4. Optimize hyperparameters with Optuna
5. Backtest on 2025 data
6. Deploy for real-time predictions

---

## 📝 License

MIT License - Free to use and modify

---

**Created**: 2026-03-20
**Version**: 10.0.0
**Status**: In Development
