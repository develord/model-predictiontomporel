# T11 - Optuna Hyperparameter Optimization Documentation
## V10 Complete Optimization Strategy

**Date**: 2026-03-20
**Status**: IN PROGRESS
**Trials**: 100 per model (200 total per crypto)

---

## Executive Summary

T11 implements comprehensive hyperparameter optimization using Optuna TPE (Tree-structured Parzen Estimator) to address the critical issues identified in T10 backtest:

**Problem**: Baseline models produce near-zero magnitude predictions
- BTC regression: MAE 4.46%, R² 0.003, avg prediction 0.04%
- ETH regression: MAE 7.12%, R² 0.033
- SOL regression: MAE 7.94%, R² -0.045

**Goal**: Improve regression accuracy to enable viable dynamic TP/SL
- Target MAE: 2.0-2.5% (BTC), 3.0-4.0% (ETH/SOL)
- Target predictions: 2-8% range (vs current 0.04-1.67%)
- Enable profitable trading after 0.2% fees

---

## Optimization Architecture

### Dual Model Optimization

Each crypto requires optimization of TWO models:

1. **Classification Model** (Direction + Confidence)
   - Objective: Maximize accuracy
   - Output: BUY (-1), HOLD (0), SELL (1)
   - Secondary: Confidence scores (probabilities)

2. **Regression Model** (Magnitude Prediction)
   - Objective: Minimize MAE
   - Output: price_target_pct (-20% to +20%)
   - Critical for dynamic TP/SL calculation

Total: **6 models** (3 cryptos × 2 models each)

### Optuna Configuration

```python
# Study setup
sampler = TPESampler(seed=42)  # Reproducible
direction = 'minimize'  # For both MAE and negative accuracy
n_trials = 100  # Per model (200 total per crypto)
```

### Search Space

All parameters optimized simultaneously:

| Parameter | Range | Type | Impact |
|-----------|-------|------|--------|
| max_depth | 3-10 | int | Tree complexity |
| learning_rate | 0.01-0.3 | log-float | Learning speed |
| n_estimators | 100-500 | int | Number of trees |
| subsample | 0.6-1.0 | float | Row sampling |
| colsample_bytree | 0.6-1.0 | float | Column sampling |
| gamma | 0-10 | float | Min split loss |
| min_child_weight | 1-10 | int | Min samples in leaf |
| reg_alpha | 0.0-2.0 | float | L1 regularization |
| reg_lambda | 0.0-2.0 | float | L2 regularization |

**Total combinations**: ~10^15 (1 quadrillion!)

---

## Implementation Details

### File Structure

```
crypto_v10_multi_tf/
├── optimization/
│   └── optuna_v10.py          # Main optimization script
├── models/
│   ├── btc_classifier_optimized.joblib
│   ├── btc_regressor_optimized.joblib
│   ├── btc_optuna_results.json
│   ├── eth_classifier_optimized.joblib
│   ├── eth_regressor_optimized.joblib
│   ├── eth_optuna_results.json
│   ├── sol_classifier_optimized.joblib
│   ├── sol_regressor_optimized.joblib
│   └── sol_optuna_results.json
└── T11_OPTUNA_OPTIMIZATION_DOCUMENTATION.md  (this file)
```

### Optimization Script (optuna_v10.py)

**Key functions**:

```python
def objective_regression(trial, X_train, y_train, X_val, y_val):
    """
    Minimize MAE for regression model

    Penalty: +10.0 if R² < -0.5 (completely broken model)
    """
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        # ... 7 more parameters
    }

    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)

    if r2 < -0.5:
        mae += 10.0  # Penalty

    return mae


def objective_classification(trial, X_train, y_train, X_val, y_val):
    """
    Minimize negative accuracy for classification model
    """
    # Similar structure, returns -accuracy
```

### Data Splits

Time-series aware splitting (no shuffling!):

```
Total samples: 3000 (BTC/ETH), 2048 (SOL)

Train: 80% (2400 / 1638)
Val:   20% (600 / 410)

Test: Separate (used in backtest, NOT in optimization)
```

**Critical**: Validation set is FUTURE data relative to training

---

## Expected Improvements

### Baseline → Optimized Comparison

#### BTC Regression
| Metric | Baseline | Target | Expected |
|--------|----------|--------|----------|
| MAE | 4.46% | 2.0-2.5% | -44-55% |
| R² | 0.003 | 0.15-0.25 | +5000-8000% |
| Pred min | 0.01% | 1.5% | +15000% |
| Pred max | 1.67% | 8.0% | +380% |
| Pred mean | 0.04% | 4.5% | +11150% |

#### ETH Regression
| Metric | Baseline | Target | Expected |
|--------|----------|--------|----------|
| MAE | 7.12% | 3.0-4.0% | -44-58% |
| R² | 0.033 | 0.20-0.30 | +500-800% |
| Pred range | narrow | 2-10% | wider |

#### SOL Regression
| Metric | Baseline | Target | Expected |
|--------|----------|--------|----------|
| MAE | 7.94% | 3.5-4.5% | -43-56% |
| R² | -0.045 | 0.15-0.25 | +430-650% |
| Pred range | narrow | 3-12% | wider |

### Classification Improvements

| Crypto | Baseline Acc | Target Acc | Expected Gain |
|--------|--------------|------------|---------------|
| BTC | 30.83% | 42-48% | +36-56% |
| ETH | 39.17% | 48-54% | +23-38% |
| SOL | 36.43% | 45-52% | +24-43% |

---

## Optimization Strategy

### TPE Sampler Advantages

1. **Smart Search**: Learns from previous trials
2. **Early Exploration**: Broad search initially
3. **Late Exploitation**: Narrow to promising regions
4. **Handles Noise**: Robust to validation variance

### Trial Budget Allocation

100 trials per model:
- **Trials 0-20**: Exploration (wide search)
- **Trials 21-60**: Balance (refine regions)
- **Trials 61-100**: Exploitation (fine-tune best)

### Convergence Indicators

Good optimization shows:
1. **Improving best value** over trials
2. **Decreasing variance** in late trials
3. **Clustering** around best params
4. **Stable final 20 trials**

---

## Parameter Analysis

### Critical Parameters (High Impact)

**learning_rate** (highest impact):
- Baseline: 0.01 (too conservative)
- Expected optimal: 0.05-0.15
- Effect: Faster learning, better convergence

**n_estimators** (second highest):
- Baseline: 100 (insufficient)
- Expected optimal: 250-400
- Effect: More complex patterns, better R²

**max_depth** (third highest):
- Baseline: 3 (too shallow)
- Expected optimal: 5-7
- Effect: Deeper trees, non-linear patterns

### Regularization Parameters

**gamma** (controls splitting):
- Baseline: 5 (maybe too high)
- Expected optimal: 3-8
- Effect: Prevents overfitting

**reg_alpha & reg_lambda** (L1/L2):
- Baseline: 1.0 each
- Expected optimal: 0.5-1.5
- Effect: Controls weight magnitudes

### Sampling Parameters

**subsample** (row sampling):
- Baseline: 0.8
- Expected optimal: 0.7-0.9
- Effect: Prevents overfitting, adds diversity

**colsample_bytree** (feature sampling):
- Baseline: 0.8
- Expected optimal: 0.7-1.0
- Effect: Feature selection per tree

---

## Post-Optimization Analysis

### Results Inspection

Each `{crypto}_optuna_results.json` contains:

```json
{
  "crypto": "btc",
  "features": 237,
  "train_samples": 2400,
  "val_samples": 600,
  "n_trials": 100,
  "regression": {
    "best_mae": 2.34,           // Best during optimization
    "final_mae": 2.35,          // Retrained on full train
    "final_r2": 0.215,
    "pred_min": 1.8,
    "pred_max": 8.2,
    "pred_mean": 4.6,
    "best_params": {
      "max_depth": 6,
      "learning_rate": 0.087,
      "n_estimators": 342,
      // ... all 9 params
    }
  },
  "classification": {
    "best_accuracy": 0.456,
    "final_accuracy": 0.453,
    "best_params": { ... }
  }
}
```

### Validation Checks

After optimization completes, verify:

1. **MAE Improvement**: ≥40% reduction
2. **R² Positive**: ≥0.15 (explains variance)
3. **Prediction Range**: 2-10% (viable for trading)
4. **Accuracy Gain**: +10-18% absolute improvement
5. **No Overfitting**: Val metrics ≈ Train metrics

---

## Impact on Backtesting

### Dynamic TP/SL Viability

**Baseline** (predicted mag = 0.04%):
```
TP = 0.04% × 0.75 = 0.03%
SL = 0.04% × 0.35 = 0.014%
Fees = 0.2%
NET = -0.17% per trade ❌
```

**Optimized** (predicted mag = 4.5%):
```
TP = 4.5% × 0.75 = 3.4%
SL = 4.5% × 0.35 = 1.6%
Fees = 0.2%
NET = +3.2% potential profit ✅
```

### Expected Backtest Improvements

| Metric | Baseline | Optimized | Change |
|--------|----------|-----------|--------|
| BTC ROI | -0.35% | +15-25% | +1500-2500% |
| ETH Trades | 0 | 50-80 | ∞ |
| SOL Trades | 0 | 40-70 | ∞ |
| Avg Win | $-0.12 | $8-15 | positive |
| Win Rate | 93.6% | 55-65% | realistic |
| Avg Hold | 1 candle | 3-7 candles | longer |

---

## Next Steps After T11

### T12: Re-run Backtest with Optimized Models

```bash
# Use optimized models instead of baseline
cd crypto_v10_multi_tf
python backtesting/backtest_v10_optimized.py
```

Expected workflow:
1. Load `*_optimized.joblib` models
2. Run backtest on test set (last 20%)
3. Compare to baseline (-0.35% ROI)
4. Document improvements

### T13: Final Parameter Tuning (if needed)

If backtest ROI < +15%:
1. Adjust TP/SL multipliers (0.75/0.35 → ?)
2. Add minimum magnitude filter (skip < 3%)
3. Optimize confidence threshold (0.40 → ?)
4. Re-run Optuna with backtest as objective

### T14: Production Deployment

Once satisfied with backtest:
1. Retrain on FULL dataset (no test split)
2. Save production models
3. Create trading bot integration
4. Implement risk management
5. Paper trading validation

---

## Troubleshooting

### Issue: MAE not improving after 50 trials

**Causes**:
- Search space too wide
- Features not informative
- Data quality issues

**Solutions**:
1. Narrow search space (reduce ranges)
2. Add feature selection
3. Check for data leakage

### Issue: R² stays negative

**Causes**:
- Model weaker than mean prediction
- Severe overfitting
- Wrong target variable

**Solutions**:
1. Increase regularization (gamma, alpha, lambda)
2. Reduce max_depth
3. Verify label generation logic

### Issue: Optimization takes >2 hours

**Expected**: ~30-45 min for 100 trials per model

**If slower**:
1. Reduce n_trials to 50
2. Use smaller datasets (sample 50%)
3. Reduce n_estimators max from 500 to 300

---

## Commands Summary

### Run Full Optimization
```bash
cd crypto_v10_multi_tf
python optimization/optuna_v10.py --crypto all --trials 100
```

### Optimize Single Crypto
```bash
python optimization/optuna_v10.py --crypto btc --trials 100
```

### Quick Test (10 trials)
```bash
python optimization/optuna_v10.py --crypto btc --trials 10
```

### Check Results
```bash
cat models/btc_optuna_results.json
cat models/eth_optuna_results.json
cat models/sol_optuna_results.json
```

---

## Performance Benchmarks

### Expected Runtime

| Task | Time | CPU Usage |
|------|------|-----------|
| Single trial (regression) | 15-20s | 100% |
| Single trial (classification) | 10-15s | 100% |
| 100 trials regression | 25-35 min | 100% |
| 100 trials classification | 17-25 min | 100% |
| **Total per crypto** | **45-60 min** | **100%** |
| **All 3 cryptos** | **2.25-3h** | **100%** |

### Memory Usage

- Peak RAM: ~4-6 GB
- Model files: ~50-150 MB each
- Results JSON: <100 KB each

---

## Success Criteria

T11 considered successful if:

### Regression Models
- [ ] BTC MAE ≤ 2.5%
- [ ] ETH MAE ≤ 4.0%
- [ ] SOL MAE ≤ 4.5%
- [ ] All R² ≥ 0.15
- [ ] Prediction ranges: 2-10%

### Classification Models
- [ ] BTC Accuracy ≥ 42%
- [ ] ETH Accuracy ≥ 48%
- [ ] SOL Accuracy ≥ 45%
- [ ] All improve ≥10% vs baseline

### Backtest Viability
- [ ] Predicted magnitudes > fees (3%+)
- [ ] TP/SL ranges viable (1-8%)
- [ ] Expected ROI > 0% on paper

---

## Conclusion

T11 Optuna optimization is the CRITICAL step that enables V10's dynamic TP/SL strategy. The baseline models demonstrated the concept works structurally, but without proper hyperparameter tuning, predictions are unusable for trading.

Expected outcome: **Transform V10 from -0.35% ROI to +15-25% ROI** through improved magnitude predictions.

**Status**: Optimization running with 100 trials per model (6 models total)
**ETA**: 2.5-3 hours for complete optimization
**Next**: T12 backtest with optimized models
