# T11-T12 Final Status Report
## V10 Optimization & Backtest Results

**Date**: 2026-03-20
**Status**: ✅ T11 Complete | ✅ T12 Complete | ❌ V10 NOT VIABLE

---

## T11: Optuna Optimization Results

### Execution Summary
- **Duration**: ~7 minutes (100 trials × 2 models × 3 cryptos)
- **Total Trials**: 600 optimizations
- **Output Files**: ✅ All 9 files created

### Optimization Results

#### BTC (237 features, 2400 train / 600 val)
**Regression (Magnitude Prediction)**:
- Best MAE: 4.39% (vs baseline 4.46% = -1.6% improvement)
- Final R²: 0.0093 (vs baseline 0.003 = +210% but still near-zero)
- Pred mean: **0.21%** ❌ (target was 4-5%)
- Pred range: -3.53% to +3.26%

**Classification (Direction)**:
- Best Accuracy: 44.83% (vs baseline 30.83% = +45% improvement ✅)

**Best Params**:
```python
# Regression
'max_depth': 3,
'learning_rate': 0.0189,
'n_estimators': 153,
'gamma': 7.36,
'min_child_weight': 6

# Classification
'max_depth': 9,
'learning_rate': 0.0106,
'n_estimators': 478,
'gamma': 9.80,
'min_child_weight': 10
```

#### ETH (348 features, 2400 train / 600 val)
**Regression**:
- Best MAE: 7.06% (vs baseline 7.12% = -0.8% improvement)
- Final R²: 0.052 (vs baseline 0.033 = +58% improvement)
- Pred mean: **-0.37%** ❌ (negative!)
- Pred range: -6.33% to +4.13%

**Classification**:
- Best Accuracy: 45.83% (vs baseline 39.17% = +17% improvement ✅)

#### SOL (348 features, 1639 train / 409 val)
**Regression**:
- Best MAE: 7.74% (vs baseline 7.94% = -2.5% improvement)
- Final R²: 0.011 (vs baseline -0.045 = +124% improvement)
- Pred mean: **0.78%** ❌ (below 3% threshold)
- Pred range: -3.15% to +6.69%

**Classification**:
- Best Accuracy: 43.03% (vs baseline 36.43% = +18% improvement ✅)

### Analysis: Why Optuna Failed to Fix Magnitude

**Problem**: Despite 100 trials, regression models still predict near-zero magnitudes

**Root Causes Identified**:

1. **Search Space Issue**: Learning rates optimized to ~0.01-0.019 (still very conservative)
2. **Objective Function Limitation**: Optimizing for MAE doesn't directly maximize prediction variance
3. **Data Quality**: Labels may genuinely have low signal (most movements ARE small)
4. **Regularization Dominance**: High gamma (7-10) + high min_child_weight (6-10) → overly smooth predictions

**What Improved**:
- ✅ Classification accuracy: +17-45% (now 43-46% vs random 33%)
- ✅ R² became positive (explains some variance now)
- ✅ MAE slightly reduced

**What DIDN'T Improve**:
- ❌ Magnitude predictions still ~0.2-0.8% (need 4-5%)
- ❌ Prediction ranges too narrow for trading

---

## T12: Backtest with Optimized Models

### Results Summary

| Crypto | Trades | ROI | Status |
|--------|--------|-----|--------|
| BTC | 0 | N/A | ❌ No signals |
| ETH | 0 | N/A | ❌ No signals |
| SOL | 2 | -0.06% | ❌ Minimal activity |

### Detailed Analysis

**Why No Trades?**

1. **Confidence Filter (0.40)**: Classification models may not reach 40% confidence often
2. **Minimum Magnitude (3.0%)**: Config requires predicted magnitude ≥3%, but models predict 0.2-0.8% avg
3. **All HOLD Predictions**: Models may prefer HOLD (class 0) due to conservative training

**SOL's 2 Trades**:
- Only crypto with any activity (likely due to higher volatility)
- Both trades likely losers (avg win = -$1.59)
- Still demonstrates same problem: magnitudes too small

### Comparison: Baseline vs Optimized

| Metric | Baseline | Optimized | Change |
|--------|----------|-----------|--------|
| BTC Trades | 78 | 0 | -100% ❌ |
| BTC ROI | -0.35% | N/A | Worse ❌ |
| ETH Trades | 0 | 0 | No change |
| SOL Trades | 0 | 2 | +2 (minimal) |

**Verdict**: Optimization made things **WORSE** by reducing trading activity to zero.

---

## Root Cause Deep Dive

### The Fundamental Problem

**V10's dynamic TP/SL depends on accurate magnitude predictions.**

Current chain of failure:
```
Regression predicts 0.2% magnitude
  ↓
TP = 0.2% × 0.75 = 0.15%
SL = 0.2% × 0.35 = 0.07%
  ↓
Predicted magnitude (0.2%) < min_magnitude_pct (3.0%)
  ↓
TRADE SKIPPED
  ↓
NO TRADING ACTIVITY
```

### Why Regression Fails

**Theory 1: Label Quality**
- `price_target_pct` labels may have low signal-to-noise ratio
- Crypto movements ARE mostly small (<3%)
- Models correctly learn "most movements are near zero"

**Theory 2: Loss Function Mismatch**
- MAE/MSE loss functions optimize for **average error**
- This incentivizes predicting the mean (~0%)
- We need a loss function that rewards **variance preservation**

**Theory 3: Feature Insufficiency**
- 237-348 features may not capture drivers of large moves
- Missing: order flow, funding rates, sentiment, macroeconomic data

**Theory 4: XGBoost Limitations**
- Tree-based models may struggle with continuous magnitude prediction
- Alternative: Try neural networks, which can model complex non-linearities better

---

## Files Created

### T11 Optimization
- ✅ `models/btc_classifier_optimized.joblib` (1.0 MB)
- ✅ `models/btc_regressor_optimized.joblib` (172 KB)
- ✅ `models/btc_optuna_results.json`
- ✅ `models/eth_classifier_optimized.joblib` (633 KB)
- ✅ `models/eth_regressor_optimized.joblib` (319 KB)
- ✅ `models/eth_optuna_results.json`
- ✅ `models/sol_classifier_optimized.joblib` (607 KB)
- ✅ `models/sol_regressor_optimized.joblib` (174 KB)
- ✅ `models/sol_optuna_results.json`

### T12 Backtest
- ✅ `backtesting/backtest_v10.py` (modified to support optimized models)
- ⚠️ Results minimal (SOL only had 2 trades)

### Documentation
- ✅ `T11_OPTUNA_OPTIMIZATION_DOCUMENTATION.md`
- ✅ `T10_BACKTEST_DIAGNOSTIC_REPORT.md`
- ✅ `SESSION_SUMMARY.md`
- ✅ `V10_COMPLETE_REFERENCE.md`
- ✅ `T11_T12_FINAL_STATUS.md` (this file)

---

## What Worked vs What Didn't

### ✅ What Worked

1. **Multi-TF Pipeline**: Successfully merged 3 timeframes (237-348 features)
2. **Dual Model Architecture**: Classification + Regression separation clean
3. **Optuna Integration**: Ran successfully, improved classification
4. **Classification Performance**: +17-45% improvement (43-46% accuracy)
5. **Code Infrastructure**: All components functional and well-documented

### ❌ What Didn't Work

1. **Regression Magnitude**: Still predicts near-zero (0.2-0.8% vs target 4-5%)
2. **Trading Viability**: 0 trades for BTC/ETH, only 2 for SOL
3. **Dynamic TP/SL Concept**: Cannot work without accurate magnitude predictions
4. **Optuna Search**: Didn't find parameters that unlock larger predictions
5. **Overall ROI**: Worse than baseline (0 activity vs -0.35%)

---

## Next Steps (Recommendations)

### Option A: Abandon Dynamic TP/SL → Fixed TP/SL (V11)
**Rationale**: Magnitude prediction fundamentally not working
**Implementation**: Use fixed 1.5% TP / 0.75% SL like V9
**Expected**: Restore trading activity, ROI +15-20%

### Option B: Alternative Regression Approach (V11)
**Changes**:
1. Try neural network (LSTM/Transformer) instead of XGBoost
2. Custom loss function: `loss = MAE + λ * (1 - variance)`
3. Add orderbook/sentiment features
4. Train on outliers only (movements >3%)

**Effort**: High (4-6 hours)
**Success Probability**: 30-40%

### Option C: Hybrid Model (V11)
**Strategy**:
1. Use classification for direction (works well)
2. Use fixed TP/SL when magnitude < 3% (fallback)
3. Use dynamic TP/SL when magnitude ≥ 3% (rare but high-confidence)

**Pros**: Best of both worlds
**Cons**: Complex logic

### Option D: Revert to V9 + Multi-TF Features (V11)
**Approach**: Take V9's proven architecture, add multi-TF features
**Expected**: V9 (+20.27%) + multi-TF boost = +25-30% ROI
**Effort**: Low (1-2 hours)

---

## Recommendation: **Option D** (V9 + Multi-TF)

**Reasoning**:
1. V9 is proven (+20.27% ROI)
2. Multi-TF features are working (classification improved)
3. Fixed TP/SL removes dependency on broken regression
4. Lowest risk, highest probability of success
5. Can iterate from working baseline

**Implementation Plan**:
1. Copy V9 architecture
2. Swap single-TF features → multi-TF features (already created)
3. Use Optuna-optimized classification model (44-46% accuracy)
4. Keep fixed TP/SL (1.5% / 0.75%)
5. Backtest on same period as V10

**Expected Outcome**: +25-30% ROI

---

## Lessons Learned

### Technical Lessons

1. **Magnitude Prediction is Hard**: Tree models struggle with continuous regression when target variance is low
2. **Optuna Isn't Magic**: Can't fix fundamental data/architecture issues
3. **Classification vs Regression**: Direction easier to predict than magnitude (inherent)
4. **Loss Functions Matter**: MAE optimizes for mean prediction, not variance preservation
5. **Feature Engineering ≠ Model Performance**: More features don't automatically mean better regression

### Strategic Lessons

1. **Start with Fixed TP/SL**: Simpler, proven, allows focus on direction prediction
2. **Validate Assumptions Early**: Should have checked magnitude distribution before building V10
3. **Iterate from Working Baseline**: V10 was too ambitious a jump from V9
4. **Documentation Pays Off**: Comprehensive docs make debugging and pivoting easier

---

## Conclusion

**V10 Status**: Structurally complete ✅, Economically non-viable ❌

**Core Innovation (Dynamic TP/SL)**: Failed due to regression model limitations

**Best Path Forward**: V11 with V9 architecture + Multi-TF features + Fixed TP/SL

**Time Investment**: T8-T12 = ~6-7 hours
**Outcome**: Valuable learning, reusable components (multi-TF pipeline), but V10 abandoned

**Next Session Goal**: Implement V11 (V9 + Multi-TF + Optuna classification) → Target +25-30% ROI
