"""
Retrain All Models for Production
===================================
Downloads latest data and retrains all 6 CNN models.
Saves models, scalers, and feature lists to LIVE_TRADING/models/

Usage:
    python retrain_all.py
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import ccxt
import json
import joblib
import logging
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR / 'scripts'))
from direction_prediction_model import CNNDirectionModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

MODEL_DIR = BASE_DIR / 'models'
MODEL_DIR.mkdir(exist_ok=True)

# Same feature engineering as BTC_PRODUCTION/02_feature_engineering.py
import ta


def download_data(crypto, timeframe, start_date):
    """Download OHLCV from Binance"""
    exchange = ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'spot'}})
    since = int(pd.Timestamp(start_date).timestamp() * 1000)
    all_ohlcv = []
    while True:
        ohlcv = exchange.fetch_ohlcv(f'{crypto}/USDT', timeframe, since, limit=1000)
        if not ohlcv:
            break
        all_ohlcv.extend(ohlcv)
        since = ohlcv[-1][0] + 1
        if len(ohlcv) < 1000:
            break
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df


def create_indicators(df, prefix=''):
    df[f'{prefix}rsi_14'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    df[f'{prefix}rsi_21'] = ta.momentum.RSIIndicator(df['close'], window=21).rsi()
    macd = ta.trend.MACD(df['close'])
    df[f'{prefix}macd_line'] = macd.macd()
    df[f'{prefix}macd_signal'] = macd.macd_signal()
    df[f'{prefix}macd_histogram'] = macd.macd_diff()
    bb = ta.volatility.BollingerBands(df['close'])
    df[f'{prefix}bb_upper'] = bb.bollinger_hband()
    df[f'{prefix}bb_middle'] = bb.bollinger_mavg()
    df[f'{prefix}bb_lower'] = bb.bollinger_lband()
    df[f'{prefix}bb_width'] = bb.bollinger_wband()
    df[f'{prefix}ema_12'] = ta.trend.EMAIndicator(df['close'], window=12).ema_indicator()
    df[f'{prefix}ema_26'] = ta.trend.EMAIndicator(df['close'], window=26).ema_indicator()
    df[f'{prefix}ema_50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
    df[f'{prefix}atr_14'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
    stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
    df[f'{prefix}stoch_k'] = stoch.stoch()
    df[f'{prefix}stoch_d'] = stoch.stoch_signal()
    df[f'{prefix}adx_14'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()
    df[f'{prefix}obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
    df[f'{prefix}cmf_20'] = ta.volume.ChaikinMoneyFlowIndicator(df['high'], df['low'], df['close'], df['volume'], window=20).chaikin_money_flow()
    df[f'{prefix}momentum_5'] = df['close'].pct_change(5)
    df[f'{prefix}momentum_10'] = df['close'].pct_change(10)
    df[f'{prefix}hist_vol_20'] = df['close'].pct_change().rolling(20).std()
    df[f'{prefix}volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    return df


def create_cross_tf(df):
    for tf1, tf2 in [('1d', '4h'), ('1d', '1w'), ('4h', '1w')]:
        c1, c2 = f'{tf1}_rsi_14', f'{tf2}_rsi_14'
        if c1 in df.columns and c2 in df.columns:
            df[f'rsi_diff_{tf1}_{tf2}'] = df[c1] - df[c2]
    rsi_cols = [c for c in df.columns if c.endswith('_rsi_14')]
    if len(rsi_cols) >= 2:
        df['rsi_bullish_count'] = sum((df[c] > 50).astype(int) for c in rsi_cols)
        df['rsi_oversold_count'] = sum((df[c] < 30).astype(int) for c in rsi_cols)
        df['rsi_overbought_count'] = sum((df[c] > 70).astype(int) for c in rsi_cols)
    macd_cols = [c for c in df.columns if c.endswith('_macd_histogram')]
    if len(macd_cols) >= 2:
        df['macd_bullish_count'] = sum((df[c] > 0).astype(int) for c in macd_cols)
    mom_cols = [c for c in df.columns if c.endswith('_momentum_5')]
    if len(mom_cols) >= 2:
        df['momentum_bullish_count'] = sum((df[c] > 0).astype(int) for c in mom_cols)
    adx_cols = [c for c in df.columns if c.endswith('_adx_14')]
    if len(adx_cols) >= 2:
        df['adx_strong_count'] = sum((df[c] > 25).astype(int) for c in adx_cols)
        df['adx_mean'] = sum(df[c] for c in adx_cols) / len(adx_cols)
    vol_cols = [c for c in df.columns if c.endswith('_hist_vol_20')]
    if len(vol_cols) >= 2:
        df['vol_mean_all_tf'] = sum(df[c] for c in vol_cols) / len(vol_cols)
    return df


def create_non_tech(df):
    df['daily_range_pct'] = (df['high'] - df['low']) / df['close']
    df['daily_range_ma5'] = df['daily_range_pct'].rolling(5).mean()
    df['daily_range_ma20'] = df['daily_range_pct'].rolling(20).mean()
    df['volatility_regime'] = (df['daily_range_ma5'] / df['daily_range_ma20']).fillna(1)
    df['volume_ma5'] = df['volume'].rolling(5).mean()
    df['volume_ma20'] = df['volume'].rolling(20).mean()
    df['volume_trend'] = (df['volume_ma5'] / df['volume_ma20']).fillna(1)
    df['price_position_20'] = (df['close'] - df['low'].rolling(20).min()) / (df['high'].rolling(20).max() - df['low'].rolling(20).min() + 1e-10)
    df['price_position_50'] = (df['close'] - df['low'].rolling(50).min()) / (df['high'].rolling(50).max() - df['low'].rolling(50).min() + 1e-10)
    body = abs(df['close'] - df['open'])
    wick = df['high'] - df['low']
    df['body_ratio'] = body / (wick + 1e-10)
    df['upper_shadow_ratio'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (wick + 1e-10)
    df['lower_shadow_ratio'] = (df[['open', 'close']].min(axis=1) - df['low']) / (wick + 1e-10)
    df['returns'] = df['close'].pct_change()
    df['consecutive_up'] = 0
    df['consecutive_down'] = 0
    for i in range(1, len(df)):
        if df.iloc[i]['returns'] > 0:
            df.iloc[i, df.columns.get_loc('consecutive_up')] = df.iloc[i-1]['consecutive_up'] + 1
        if df.iloc[i]['returns'] < 0:
            df.iloc[i, df.columns.get_loc('consecutive_down')] = df.iloc[i-1]['consecutive_down'] + 1
    df['distance_from_sma20'] = (df['close'] / df['close'].rolling(20).mean() - 1)
    df['distance_from_sma50'] = (df['close'] / df['close'].rolling(50).mean() - 1)
    df['higher_highs'] = (df['high'] > df['high'].shift(1)).rolling(5).sum()
    df['lower_lows'] = (df['low'] < df['low'].shift(1)).rolling(5).sum()
    df['trend_score'] = df['higher_highs'] - df['lower_lows']
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_month'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    return df


def create_labels(df, tp_pct=0.015, sl_pct=0.0075):
    labels = []
    for i in range(len(df)):
        if i >= len(df) - 1:
            labels.append(-1)
            continue
        entry = df.iloc[i]['close']
        tp, sl = entry * (1 + tp_pct), entry * (1 - sl_pct)
        hit_tp, hit_sl = False, False
        for j in range(i+1, min(i+11, len(df))):
            if df.iloc[j]['high'] >= tp:
                hit_tp = True
                break
            if df.iloc[j]['low'] <= sl:
                hit_sl = True
                break
        if hit_tp:
            labels.append(1)
        elif hit_sl:
            labels.append(0)
        else:
            labels.append(-1)
    return np.array(labels)


def build_features(crypto, start_date):
    """Full pipeline: download + feature engineering"""
    logger.info(f"\n{'='*50}")
    logger.info(f"Building features for {crypto}")

    # Download
    df_1d = download_data(crypto, '1d', start_date)
    logger.info(f"  1d: {len(df_1d)} candles")

    df_1d = create_indicators(df_1d, '1d_')

    for tf in ['4h', '1w']:
        df_tf = download_data(crypto, tf, start_date)
        logger.info(f"  {tf}: {len(df_tf)} candles")
        df_tf = create_indicators(df_tf, f'{tf}_')
        tf_cols = ['date'] + [c for c in df_tf.columns if c.startswith(f'{tf}_')]
        df_1d = pd.merge_asof(df_1d.sort_values('date'), df_tf[tf_cols].sort_values('date'), on='date', direction='backward')

    df_1d = create_cross_tf(df_1d)
    df_1d = create_non_tech(df_1d)
    df_1d['label'] = create_labels(df_1d)

    exclude = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'date', 'label', 'returns']
    feature_cols = [c for c in df_1d.columns if c not in exclude and not c.startswith('Unnamed')]
    for c in feature_cols:
        df_1d[c] = pd.to_numeric(df_1d[c], errors='coerce')
    df_1d[feature_cols] = df_1d[feature_cols].replace([np.inf, -np.inf], np.nan)

    logger.info(f"  Features: {len(feature_cols)} | Rows: {len(df_1d)}")
    return df_1d, feature_cols


def train_cnn(crypto, df, feature_cols, seq_len=30, epochs=200, patience=35):
    """Train CNN model on all data"""
    logger.info(f"Training {crypto} CNN...")

    # Prepare data
    X_raw = df[feature_cols].fillna(0).values.astype(np.float32)
    y = df['label'].values

    scaler = RobustScaler()
    X = np.clip(np.nan_to_num(scaler.fit_transform(X_raw), nan=0, posinf=0, neginf=0), -5, 5).astype(np.float32)

    # Build sequences
    seqs, labels = [], []
    for i in range(seq_len, len(X)):
        if y[i] != -1:
            seqs.append(X[i-seq_len:i])
            labels.append(y[i])
    seqs, labels = np.array(seqs), np.array(labels)

    n_sl, n_tp = (labels == 0).sum(), (labels == 1).sum()
    logger.info(f"  Sequences: {len(seqs)} (SL={n_sl}, TP={n_tp})")

    # Augment
    aug_X = [seqs]
    aug_y = [labels]
    for _ in range(2):
        aug_X.append(seqs + np.random.normal(0, 0.015, seqs.shape).astype(np.float32))
        aug_y.append(labels)
    aug_X, aug_y = np.concatenate(aug_X), np.concatenate(aug_y)

    # Split last 20% for validation
    split = int(len(seqs) * 0.8)
    train_X, train_y = torch.FloatTensor(aug_X), torch.LongTensor(aug_y)
    val_X, val_y = torch.FloatTensor(seqs[split:]), torch.LongTensor(labels[split:])

    train_loader = DataLoader(TensorDataset(train_X, train_y), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_X, val_y), batch_size=64)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feat_dim = len(feature_cols)
    model = CNNDirectionModel(feature_dim=feat_dim, sequence_length=seq_len, dropout=0.4).to(device)

    w_sl = len(labels) / (2 * n_sl) if n_sl > 0 else 1.0
    w_tp = len(labels) / (2 * n_tp) if n_tp > 0 else 1.0
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([w_sl, w_tp]).to(device), label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=0.0015, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)

    best_acc, best_epoch, patience_cnt = 0, 0, 0
    model_path = MODEL_DIR / f'{crypto.lower()}_cnn_model.pt'

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
        scheduler.step(epoch)

        # Validate
        model.eval()
        correct, total = 0, 0
        n_buy, n_sell = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).argmax(1)
                correct += (pred == yb).sum().item()
                total += yb.size(0)
                n_buy += (pred == 1).sum().item()
                n_sell += (pred == 0).sum().item()

        acc = correct / total if total > 0 else 0
        if acc > best_acc and n_buy >= 5 and n_sell >= 5:
            best_acc = acc
            best_epoch = epoch + 1
            patience_cnt = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'feature_dim': feat_dim,
                'sequence_length': seq_len,
                'model_type': 'cnn',
                'epoch': epoch + 1,
                'val_acc': acc,
            }, model_path)
        else:
            patience_cnt += 1

        if (epoch + 1) % 10 == 0:
            logger.info(f"  E{epoch+1}: acc={acc:.3f} BUY={n_buy} SELL={n_sell} best={best_acc:.3f}")

        if patience_cnt >= patience:
            break

    logger.info(f"  Best epoch: {best_epoch} | Best acc: {best_acc:.3f}")

    # Save scaler and features
    joblib.dump(scaler, MODEL_DIR / f'{crypto.lower()}_feature_scaler.joblib')
    with open(MODEL_DIR / f'{crypto.lower()}_features.json', 'w') as f:
        json.dump(feature_cols, f)

    return model_path


# Coin configs
COIN_CONFIGS = {
    'BTC':  {'start': '2017-01-01'},
    'ETH':  {'start': '2018-01-01'},
    'SOL':  {'start': '2020-08-01'},
    'DOGE': {'start': '2019-07-01'},
    'XRP':  {'start': '2018-01-01'},
    'AVAX': {'start': '2020-09-01'},
}


def main():
    logger.info("=" * 70)
    logger.info("RETRAINING ALL MODELS FOR PRODUCTION")
    logger.info("=" * 70)

    for crypto, cfg in COIN_CONFIGS.items():
        try:
            df, feature_cols = build_features(crypto, cfg['start'])
            model_path = train_cnn(crypto, df, feature_cols)
            logger.info(f"{crypto}: Model saved to {model_path}")
        except Exception as e:
            logger.error(f"{crypto}: FAILED - {e}")

    logger.info("\n" + "=" * 70)
    logger.info("ALL MODELS RETRAINED")
    logger.info("=" * 70)
    logger.info(f"Models saved to: {MODEL_DIR}")


if __name__ == "__main__":
    main()
