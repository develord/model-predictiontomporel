# BTC Production Trading System

Complete PyTorch-based trading system for Bitcoin with Transformer+LSTM architecture.

## Performance (Q1 2026 Backtest)

| Metric | Value |
|--------|-------|
| Total Return | **+4.81%** |
| Trades | 42 |
| Win Rate | ~52% |
| Initial Capital | $1,000 |
| Final Capital | $1,048.10 |

## System Overview

### Model Architecture
- **Algorithm**: PyTorch (Transformer + LSTM + Attention)
- **Features**: 90 enhanced technical indicators
- **Sequence Length**: 60 days
- **Training Period**: 2018-2024
- **Test Period**: 2026-01-01 to 2026-03-24 (Q1 2026)
- **Model Size**: 17 MB

### Trading Parameters
- **TP (Take Profit)**: 1.5%
- **SL (Stop Loss)**: 0.75%
- **Position Size**: 95%
- **Trading Fee**: 0.1%
- **Slippage**: 0.05%

## Folder Structure

```
BTC_PRODUCTION/
├── 01_download_data.py          # Downloads 1d BTC data from Binance
├── 02_feature_engineering.py    # Creates 90 enhanced features
├── 03_train_pytorch_model.py    # Model info (already trained)
├── 04_backtest.py               # Backtests on Q1 2026
├── 05_production_inference.py   # Real-time predictions
├── data/cache/                  # Raw data and features
│   ├── btc_1d_data.csv
│   └── btc_features.csv
├── models/                      # PyTorch models
│   └── BTC_direction_model.pt   # Trained model (17MB)
├── scripts/                     # Helper scripts
│   ├── direction_prediction_model.py      # PyTorch architecture
│   └── enhanced_features_fixed.py         # Feature engineering
├── results/                     # Backtest results
└── README.md                    # This file
```

## Usage

### 1. Download Data
```bash
python 01_download_data.py
```
Downloads 1d OHLCV data from Binance since 2018-01-01.

### 2. Create Features
```bash
python 02_feature_engineering.py
```
Generates 90 enhanced technical features using EnhancedFeatureEngineering.

### 3. Check Model (Optional)
```bash
python 03_train_pytorch_model.py
```
Shows model info. Model is already trained (BTC_direction_model.pt).

### 4. Backtest
```bash
python 04_backtest.py
```
Runs backtest on Q1 2026 data.

### 5. Production Inference
```bash
python 05_production_inference.py
```
Makes real-time predictions using latest market data.

## Technical Features (90 Total)

The model uses 90 enhanced features across multiple categories:

### Price Action
- Multiple timeframe returns (1d, 3d, 5d, 7d, 14d, 21d, 30d)
- Rolling highs/lows (7d, 14d, 30d, 60d)
- Price volatility metrics

### Momentum Indicators
- RSI (14, 21 periods)
- Stochastic Oscillator (K, D)
- MACD (line, signal, histogram)
- Rate of Change (ROC)

### Trend Indicators
- EMAs (12, 26, 50, 200 periods)
- SMA crossovers
- ADX (14 period)
- Ichimoku components

### Volatility
- Bollinger Bands (upper, middle, lower, width)
- ATR (14 period)
- Historical volatility
- Keltner Channels

### Volume
- OBV (On-Balance Volume)
- CMF (Chaikin Money Flow)
- Volume ratios
- Volume moving averages

### Advanced Features
- Market structure (higher highs, lower lows)
- Support/resistance levels
- Fibonacci retracements
- Pattern recognition features

## Model Details

### PyTorch Architecture

```python
DirectionPredictionModel:
- Feature Extractor: Linear(500, 256) + LayerNorm + GELU
- Positional Encoding: Sinusoidal
- Transformer Encoder: 4 layers, 8 heads, d_model=256
- LSTM: 2 layers, hidden_dim=128, bidirectional
- Attention: Multi-head attention (4 heads)
- Classifier: Linear layers (256 → 128 → 64 → 2)
- Output: Binary classification (SHORT=0, LONG=1)
```

### Training Details
- **Dataset**: 2018-2024 BTC 1d data
- **Sequence Length**: 60 days
- **Features**: 500 (90 base features + engineered)
- **Loss Function**: CrossEntropyLoss
- **Optimizer**: AdamW
- **Learning Rate**: 1e-4
- **Batch Size**: 32

## Results Interpretation

### Performance Comparison
- **BTC PyTorch**: +4.81% (Q1 2026, 42 trades, 52% WR)
- **ETH XGBoost**: +6.24% (Q1 2026, 1 trade, 100% WR)
- **SOL XGBoost**: +6.24% (Q1 2026, 1 trade, 100% WR)

### Why Lower Return vs ETH/SOL?
1. **More trades**: 42 trades vs 1 trade (more exposure to fees/slippage)
2. **Lower win rate**: 52% vs 100% (less selective)
3. **Different strategy**: Active trading vs patient waiting
4. **Market conditions**: Q1 2026 was choppy for BTC

### Advantages of PyTorch Model
- Captures complex temporal patterns
- Learns market regime changes
- Can be fine-tuned continuously
- Handles non-linear relationships

## Production Deployment

To deploy this system:

1. **Data Pipeline**: Schedule `01_download_data.py` daily
2. **Feature Update**: Run `02_feature_engineering.py` after data download
3. **Inference**: Run `05_production_inference.py` for signals
4. **Monitoring**: Track predictions vs actual outcomes
5. **Retraining**: Retrain model quarterly with new data

## Dependencies

```bash
pip install ccxt pandas numpy torch ta matplotlib
```

Specific versions:
- torch >= 1.12
- pandas >= 1.5
- numpy >= 1.23
- ccxt >= 3.0

## Risk Management

- Never risk more than 2% per trade
- Use strict TP/SL levels
- Monitor model confidence scores
- Stop trading during extreme volatility
- Keep 5% cash buffer

## Notes

- Model processes sequences of 60 days
- Predictions update daily (1d timeframe)
- Confidence threshold: >60% for trade signals
- Model works best in trending markets
- Struggles during low-volatility periods

## Potential Improvements

1. **Ensemble with XGBoost**: Combine PyTorch + XGBoost predictions
2. **Dynamic TP/SL**: Adjust based on volatility
3. **Market regime detection**: Different strategies for different regimes
4. **Multi-asset learning**: Train on BTC+ETH+SOL together
5. **Reinforcement learning**: Learn optimal trade timing

## Support

For issues or questions, refer to main documentation: `README_PRODUCTION_MODELS.md`

---

**Version**: 1.0
**Last Updated**: 29 March 2026
**Status**: Production Ready
**Model Type**: PyTorch Deep Learning
