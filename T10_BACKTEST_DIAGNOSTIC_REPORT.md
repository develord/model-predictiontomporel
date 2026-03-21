# T10 Backtest Diagnostic Report
## V10 Baseline - Dynamic TP/SL Issues

**Date**: 2026-03-20
**Status**: CRITICAL ISSUES IDENTIFIED

---

## Executive Summary

V10 backtest completed with **MAJOR PROBLEMS**:
- **BTC**: -0.35% ROI (WORSE than V9 +20.27%)
- **ETH/SOL**: NO TRADES EXECUTED (confidence filter too strict)
- **Root Cause**: Regression model predicting near-zero magnitudes

---

## Issue 1: Regression Model Performance

### Problem
The regression models are barely making predictions:

| Crypto | Avg Predicted Magnitude | R² Score | MAE |
|--------|-------------------------|----------|-----|
| BTC    | +0.04%                 | 0.003    | 4.46% |
| ETH    | Unknown (no trades)    | 0.033    | 7.12% |
| SOL    | Unknown (no trades)    | -0.045   | 7.94% |

**Analysis:**
- Models regressing to mean (predicting ~0%)
- Very conservative parameters (from V9 anti-overfitting)
- Only 100 estimators, lr=0.01, max_depth=3
- NO tuning performed yet

### Impact on Trading
With magnitude = 0.04%:
- TP = 0.04% × 0.75 = **0.03%** (3 basis points!)
- SL = 0.04% × 0.35 = **0.014%** (1.4 basis points!)

Result: TP/SL levels are hit on the SAME candle → all trades exit after 1 period

---

## Issue 2: BTC Results (-0.35% ROI)

### Trading Performance
```
Total Trades:    78
Wins:            73 (93.6%)
Losses:          5
Avg Win:         $-0.12   ← NEGATIVE despite "winning"!
Avg Loss:        $-5.23
Profit Factor:   -0.33
Sharpe Ratio:    -3.17
Avg Hold:        1.0 candles
```

### Paradox Analysis
How can 93.6% win rate produce negative ROI?

**Sample Trades:**
```csv
Trade 5: SELL, magnitude=0.17%, TP hit, pnl_pct=+0.07%, pnl=-$0.72
Trade 6: SELL, magnitude=0.20%, TP hit, pnl_pct=+0.05%, pnl=-$0.53
Trade 13: SELL, magnitude=0.08%, TP hit, pnl_pct=-0.14%, pnl=-$1.38
```

**Root Cause:**
1. Predicted magnitude too small (0.04-1.67%)
2. TP/SL calculated from magnitude (TP=0.75×mag, SL=0.35×mag)
3. TP levels are SMALLER than fee structure (0.2% = 0.1% × 2)
4. Even "winning" trades lose money after fees
5. Position size only 10% of capital → tiny absolute PnL

**Fee Impact:**
- Fee per trade: 0.1% entry + 0.1% exit = 0.2% total
- TP target: 0.03% (for mag=0.04%)
- **Net result**: -0.17% per "win"!

---

## Issue 3: ETH/SOL - No Trades

### Error Message
```
Error with eth: string indices must be integers
Error with sol: string indices must be integers
```

### Root Cause
1. `_calculate_metrics()` returns `{'error': 'No trades executed'}` when trades list is empty
2. `run_backtest_for_crypto()` expects tuple `(metrics, trades_df)`
3. Type mismatch causes crash

### Why No Trades?
Checking confidence threshold (0.40):
- ETH/SOL models may not reach 0.40 confidence on any predictions
- Or: All predictions are HOLD (class 0)
- Or: Predicted magnitudes below min_magnitude_pct (3.0%)

**Need to investigate:**
```python
# Check ETH/SOL prediction distribution
print(classifier.predict(X_test))  # Are all predictions class 0?
print(classifier.predict_proba(X_test).max(axis=1))  # Max confidences?
```

---

## Technical Analysis

### Dynamic TP/SL Logic (backtest_v10.py:93-114)
```python
if pred_class == 1:  # BUY
    tp_pct = abs(predicted_magnitude) * 0.75
    sl_pct = abs(predicted_magnitude) * 0.35
    tp_price = entry_price * (1 + tp_pct / 100)
    sl_price = entry_price * (1 - sl_pct / 100)
```

### Example Calculation
```
Entry: $100,000
Predicted Magnitude: +0.04%

TP = 0.04 × 0.75 = 0.03%
TP Price = $100,000 × 1.0003 = $100,030

SL = 0.04 × 0.35 = 0.014%
SL Price = $100,000 × 0.99986 = $99,986

Range: $99,986 - $100,030 ($44 range on $100k!)
```

**Problem:** $44 range is SMALLER than typical bid-ask spread + slippage!

---

## Comparison: V9 vs V10 Baseline

| Metric | V9 Final | V10 Baseline (BTC only) |
|--------|----------|-------------------------|
| ROI | +20.27% | **-0.35%** |
| Win Rate | ~55-60% | 93.6% |
| Trades | ~200-300 | 78 |
| Avg Hold | 3-7 candles | 1 candle |
| TP/SL | Fixed (±1.5%) | Dynamic (0.01-1.67%) |

**Verdict:** V10 baseline is WORSE than V9 due to undertrained regression

---

## Root Cause Summary

### 1. Undertrained Regression Model
- Only 100 trees (vs V9 optimized 200-500)
- lr = 0.01 (very conservative)
- max_depth = 3 (shallow trees)
- No hyperparameter optimization yet

### 2. Inappropriate Fee Structure
- 0.2% total fees (0.1% × 2)
- TP targets averaging 0.03-0.50%
- **Fees > Profit Target** in most cases

### 3. Missing Minimum Magnitude Filter
Config has `min_magnitude_pct: 3.0` but NOT enforced in backtest logic

Current code:
```python
if confidence < self.config['confidence_threshold']:
    continue  # Filter by confidence

# MISSING: Filter by magnitude!
# if abs(predicted_magnitude) < self.config['min_magnitude_pct']:
#     continue
```

---

## Action Items

### Priority 1: Fix Immediate Bugs
- [X] JSON serialization (float32 → float)
- [ ] Fix ETH/SOL "string indices" error
- [ ] Add magnitude filter (min 3%)
- [ ] Adjust fee structure or TP multipliers

### Priority 2: Improve Regression Model (T11 - Optuna)
- [ ] Increase n_estimators: 100 → 300+
- [ ] Tune learning_rate: 0.01 → 0.05-0.15
- [ ] Increase max_depth: 3 → 5-8
- [ ] Optimize gamma, subsample, colsample

### Priority 3: TP/SL Strategy Refinement
Two options:

**Option A: Minimum TP/SL**
```python
tp_pct = max(abs(predicted_magnitude) * 0.75, 1.0)  # At least 1%
sl_pct = max(abs(predicted_magnitude) * 0.35, 0.5)  # At least 0.5%
```

**Option B: Adjust Multipliers**
```python
tp_pct = abs(predicted_magnitude) * 1.5  # Increase from 0.75
sl_pct = abs(predicted_magnitude) * 0.5  # Increase from 0.35
```

---

## Expected Improvements After T11

### Regression MAE Target
| Crypto | Current MAE | Target MAE | Improvement |
|--------|-------------|------------|-------------|
| BTC | 4.46% | 2.0-2.5% | -45-55% |
| ETH | 7.12% | 3.0-4.0% | -44-58% |
| SOL | 7.94% | 3.5-4.5% | -43-56% |

### Predicted Magnitude Range
- **Current**: 0.04-1.67% (avg 0.4%)
- **Target**: 2.0-8.0% (avg 4.5%)
- **Impact**: TP = 3.4%, SL = 1.6% (viable with fees)

### ROI Projection
Assuming magnitude improves to 4.5% avg:
- TP = 4.5 × 0.75 = 3.4% (beats 0.2% fees)
- Win rate ~65% (more reasonable)
- Expected ROI: **+15-25%** (still below V9 without further tuning)

---

## Next Steps

1. **Fix ETH/SOL crash** (return tuple from _calculate_metrics)
2. **Add magnitude filter** (skip trades < 3%)
3. **Run T11 Optuna** to improve regression
4. **Re-run backtest** with optimized models
5. **Adjust TP/SL multipliers** if still underperforming

---

## Conclusion

V10 baseline backtest reveals that **dynamic TP/SL depends critically on accurate magnitude predictions**. The current regression models (using V9 conservative parameters without tuning) produce predictions too small to overcome transaction costs.

**T11 (Optuna optimization) is CRITICAL** - without it, V10 will underperform V9.

The multi-timeframe features are working (237-348 features), but the dual-model system needs proper hyperparameter tuning to unlock its potential.

**Status**: T10 technically complete (backtest runs), but results show V10 baseline is NOT viable. Must proceed to T11 immediately.
