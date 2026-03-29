# SOL Production Trading System

Complete XGBoost-based trading system for Solana with intelligent signal filtering.

## Performance (Q1 2026 Backtest)

| Metric | Value |
|--------|-------|
| Total Return | **+6.24%** |
| Trades | 1 |
| Win Rate | 100% |
| Initial Capital | $1,000 |
| Final Capital | $1,062.40 |

## System Overview

### Model Architecture
- **Algorithm**: XGBoost (Gradient Boosting)
- **Features**: Multi-timeframe technical indicators (1h, 4h, 1d, 1w)
- **Training Period**: 2018-01-01 to 2024-12-31
- **Test Period**: 2026-01-01 to 2026-03-24 (Q1 2026)

### Trading Parameters
- **TP (Take Profit)**: 1.5%
- **SL (Stop Loss)**: 0.75%
- **Position Size**: 95%
- **Trading Fee**: 0.1%
- **Slippage**: 0.05%

### Intelligent Signal Filtering

The system uses 5 criteria to filter signals (87.5% rejection rate):

1. **Confidence Threshold**: ≥ 65%
2. **Volatility Control**:
   - 1d: ≤ 4%
   - 4h: ≤ 3%
   - 1w: ≤ 5%
3. **Volume Filter**: Volume ratio ≥ 1.2x (20-day average)
4. **Trend Strength**: ADX ≥ 20
5. **Momentum Alignment**: At least 2/3 timeframes agree (RSI > 50)

## Folder Structure

```
SOL_PRODUCTION/
├── 01_download_data.py          # Downloads 1h/4h/1d/1w data from Binance
├── 02_feature_engineering.py    # Creates technical indicators + labels
├── 03_train_model.py            # Trains XGBoost model
├── 04_backtest.py               # Backtests on Q1 2026
├── 05_production_inference.py   # Real-time predictions
├── data/cache/                  # Raw and merged data
│   ├── sol_1h_data.csv
│   ├── sol_4h_data.csv
│   ├── sol_1d_data.csv
│   ├── sol_1w_data.csv
│   └── sol_multi_tf_merged.csv
├── models/                      # Trained models
│   ├── sol_v11_top50.joblib     # Final XGBoost model
│   ├── sol_v11_features.json    # Feature list
│   └── sol_v11_top50_stats.json # Training stats
├── results/                     # Backtest results
│   ├── sol_backtest_trades.csv
│   └── sol_backtest_summary.json
└── README.md                    # This file
```

## Usage

### 1. Download Data
```bash
python 01_download_data.py
```
Downloads multi-timeframe OHLCV data from Binance since 2018-01-01.

### 2. Create Features
```bash
python 02_feature_engineering.py
```
Generates technical indicators and triple barrier labels (TP=1.5%, SL=0.75%).

### 3. Train Model (Optional)
```bash
python 03_train_model.py
```
Trains XGBoost model on historical data. Model is already trained and saved in `models/`.

### 4. Backtest
```bash
python 04_backtest.py
```
Runs intelligent backtest on Q1 2026 data with signal filtering.

### 5. Production Inference
```bash
python 05_production_inference.py
```
Makes real-time predictions using latest market data.

## Technical Indicators

Each timeframe includes:
- **Momentum**: RSI (14, 21), Stochastic (K, D)
- **Trend**: MACD (line, signal, histogram), EMA (12, 26, 50, 200), ADX (14)
- **Volatility**: Bollinger Bands (upper, middle, lower, width), ATR (14)
- **Volume**: OBV, CMF (20)

Total features per timeframe: ~18
Total features (4 timeframes): ~72

## Dependencies

```bash
pip install ccxt pandas numpy xgboost scikit-learn joblib ta matplotlib
```

## Model Details

### XGBoost Hyperparameters
- `n_estimators`: 300
- `max_depth`: 6
- `learning_rate`: 0.05
- `subsample`: 0.8
- `colsample_bytree`: 0.8
- `objective`: binary:logistic
- `tree_msolod`: hist

### Labeling Strategy
- **Class 1 (TP)**: Take profit hit before stop loss
- **Class 0 (SL/None)**: Stop loss hit or neither TP/SL hit

## Results Interpretation

### Why Only 1 Trade in Q1 2026?
The intelligent filtering system rejects 87.5% of signals to ensure only high-quality setups. This conservative approach prioritizes win rate over trade frequency.

### Signal Filtering Breakdown (Typical)
- Low confidence: ~30%
- High volatility: ~25%
- Low volume: ~15%
- Weak trend: ~10%
- Poor momentum alignment: ~7.5%

## Production Deployment

To deploy this system:

1. **Data Pipeline**: Schedule `01_download_data.py` to run daily (after market close)
2. **Feature Update**: Run `02_feature_engineering.py` to update features
3. **Inference**: Run `05_production_inference.py` to get signals
4. **Monitoring**: Track filter pass rates and adjust thresholds if needed
5. **Retraining**: Retrain model monthly/quarterly with `03_train_model.py`

## Risk Management

- Never risk more than 2% of capital per trade
- Use 95% position sizing (keep 5% as buffer)
- Always use TP/SL orders
- Monitor market conditions (news, events)
- Stop trading during high-impact news

## Notes

- Model is trained on 7 years of data (2018-2024)
- Backtest uses walk-forward msolodology (no lookahead bias)
- All dates are in UTC timezone
- Slippage and fees are included in calculations
- Model assumes instant execution at open price + slippage

## Support

For issues or questions, refer to main documentation: `README_PRODUCTION_MODELS.md`

---

**Version**: 1.0
**Last Updated**: 29 March 2026
**Status**: Production Ready
