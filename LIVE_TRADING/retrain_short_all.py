"""
Retrain ALL SHORT Models - Walk-Forward Validation
====================================================
Train on full history, validate on 2025 H1, test on 2025 H2.
Uses simple CNNDirectionModel (not DeepCNN) for stability.
Adds bear-specific features.

Usage:
    python retrain_short_all.py
"""

import sys
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
import ta
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR / 'scripts'))
from direction_prediction_model import CNNDirectionModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

MODEL_DIR = BASE_DIR / 'models'
MODEL_DIR.mkdir(exist_ok=True)

# SHORT params (balanced)
SHORT_TP = 0.020  # 2% drop
SHORT_SL = 0.010  # 1% rise
SEQ_LEN = 30      # Standard (not 45 - simpler model)
BATCH = 64
EPOCHS = 200
LR = 0.0015
PATIENCE = 35


def download_data(crypto, timeframe, start):
    ex = ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'spot'}})
    since = int(pd.Timestamp(start).timestamp() * 1000)
    all_ohlcv = []
    while True:
        ohlcv = ex.fetch_ohlcv(f'{crypto}/USDT', timeframe, since, limit=1000)
        if not ohlcv: break
        all_ohlcv.extend(ohlcv)
        since = ohlcv[-1][0] + 1
        if len(ohlcv) < 1000: break
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


def add_bear_features(df):
    for w in [10, 20, 50]:
        sma = df['close'].rolling(w).mean()
        df[f'price_above_sma{w}_pct'] = (df['close'] / sma - 1) * 100
    df['roc_5'] = df['close'].pct_change(5) * 100
    df['roc_10'] = df['close'].pct_change(10) * 100
    df['roc_deceleration'] = df['roc_5'] - df['roc_10']
    df['price_change_5'] = df['close'].pct_change(5)
    df['vol_change_5'] = df['volume'].pct_change(5)
    df['vol_price_divergence'] = np.where(
        (df['price_change_5'] > 0) & (df['vol_change_5'] < 0), 1,
        np.where((df['price_change_5'] < 0) & (df['vol_change_5'] > 0), -1, 0))
    df['is_red'] = (df['close'] < df['open']).astype(int)
    df['consec_red'] = 0
    for i in range(1, len(df)):
        if df.iloc[i]['is_red']:
            df.iloc[i, df.columns.get_loc('consec_red')] = df.iloc[i-1]['consec_red'] + 1
    df['high_rejection'] = (df['high'] - df['close']) / (df['high'] - df['low'] + 1e-10)
    df['dist_from_high_20'] = (df['close'] / df['high'].rolling(20).max() - 1) * 100
    df['dist_from_high_50'] = (df['close'] / df['high'].rolling(50).max() - 1) * 100
    if '1d_atr_14' in df.columns:
        df['vol_expansion'] = df['1d_atr_14'] / df['1d_atr_14'].shift(5)
    if '1d_rsi_14' in df.columns:
        df['rsi_slope_5'] = df['1d_rsi_14'].diff(5)
        df['price_slope_5'] = df['close'].pct_change(5) * 100
        df['rsi_price_divergence'] = np.where(
            (df['price_slope_5'] > 0) & (df['rsi_slope_5'] < 0), 1,
            np.where((df['price_slope_5'] < 0) & (df['rsi_slope_5'] > 0), -1, 0))
    # Cross-TF and non-tech
    df['daily_range_pct'] = (df['high'] - df['low']) / df['close']
    df['volatility_regime'] = (df['daily_range_pct'].rolling(5).mean() / df['daily_range_pct'].rolling(20).mean()).fillna(1)
    df['distance_from_sma20'] = (df['close'] / df['close'].rolling(20).mean() - 1)
    df['distance_from_sma50'] = (df['close'] / df['close'].rolling(50).mean() - 1)
    df['trend_score'] = (df['high'] > df['high'].shift(1)).rolling(5).sum() - (df['low'] < df['low'].shift(1)).rolling(5).sum()
    return df


def create_short_labels(df):
    labels = []
    for i in range(len(df)):
        if i >= len(df) - 1: labels.append(-1); continue
        entry = df.iloc[i]['close']
        tp, sl = entry * (1 - SHORT_TP), entry * (1 + SHORT_SL)
        hit_tp, hit_sl = False, False
        for j in range(i+1, min(i+11, len(df))):
            if df.iloc[j]['low'] <= tp: hit_tp = True; break
            if df.iloc[j]['high'] >= sl: hit_sl = True; break
        if hit_tp: labels.append(1)
        elif hit_sl: labels.append(0)
        else: labels.append(-1)
    return np.array(labels)


def build_features(crypto, start):
    logger.info(f"\n{'='*50}\nBuilding features for {crypto}")
    df_1d = download_data(crypto, '1d', start)
    logger.info(f"  1d: {len(df_1d)} candles")
    df_1d = create_indicators(df_1d, '1d_')
    for tf in ['4h', '1w']:
        df_tf = download_data(crypto, tf, start)
        df_tf = create_indicators(df_tf, f'{tf}_')
        tf_cols = ['date'] + [c for c in df_tf.columns if c.startswith(f'{tf}_')]
        df_1d = pd.merge_asof(df_1d.sort_values('date'), df_tf[tf_cols].sort_values('date'), on='date', direction='backward')
    df_1d = add_bear_features(df_1d)
    df_1d['label'] = create_short_labels(df_1d)
    exclude = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'date', 'label',
               'returns', 'is_red', 'price_change_5', 'vol_change_5']
    feature_cols = [c for c in df_1d.columns if c not in exclude and not c.startswith('Unnamed')]
    for c in feature_cols:
        df_1d[c] = pd.to_numeric(df_1d[c], errors='coerce')
    df_1d[feature_cols] = df_1d[feature_cols].replace([np.inf, -np.inf], np.nan)
    logger.info(f"  Features: {len(feature_cols)} | Rows: {len(df_1d)}")
    n1 = (df_1d['label'] == 1).sum()
    n0 = (df_1d['label'] == 0).sum()
    logger.info(f"  Labels: SHORT={n1} NO={n0} ratio={n1/(n0+1):.2f}")
    return df_1d, feature_cols


def train_short(crypto, df, feature_cols):
    logger.info(f"Training {crypto} SHORT (walk-forward)...")

    # Walk-forward: Train < 2025, Val 2025-H1, Test 2025-H2
    train_df = df[df['date'] < '2025-01-01'].copy()
    val_df = df[(df['date'] >= '2025-01-01') & (df['date'] < '2025-07-01')].copy()
    test_df = df[df['date'] >= '2025-07-01'].copy()

    logger.info(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # Scale
    X_train = train_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    X_val = val_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    X_test = test_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)

    scaler = RobustScaler()
    X_train_s = np.clip(np.nan_to_num(scaler.fit_transform(X_train), nan=0, posinf=0, neginf=0), -5, 5).astype(np.float32)
    X_val_s = np.clip(np.nan_to_num(scaler.transform(X_val), nan=0, posinf=0, neginf=0), -5, 5).astype(np.float32)
    X_test_s = np.clip(np.nan_to_num(scaler.transform(X_test), nan=0, posinf=0, neginf=0), -5, 5).astype(np.float32)

    y_train = train_df['label'].values
    y_val = val_df['label'].values
    y_test = test_df['label'].values

    # Build sequences
    def build_seq(X, y):
        s, l = [], []
        for i in range(SEQ_LEN, len(X)):
            if y[i] != -1: s.append(X[i-SEQ_LEN:i]); l.append(y[i])
        return np.array(s), np.array(l)

    train_seqs, train_labels = build_seq(X_train_s, y_train)
    val_seqs, val_labels = build_seq(X_val_s, y_val)
    test_seqs, test_labels = build_seq(X_test_s, y_test)

    n0, n1 = (train_labels == 0).sum(), (train_labels == 1).sum()
    logger.info(f"  Seqs: train={len(train_seqs)} (NO={n0} SHORT={n1}) val={len(val_seqs)} test={len(test_seqs)}")

    if len(train_seqs) < 100 or len(val_seqs) < 20:
        logger.error(f"  {crypto}: Not enough data")
        return None

    # Augment training (3x)
    aug_X = np.concatenate([train_seqs] + [train_seqs + np.random.normal(0, 0.015, train_seqs.shape).astype(np.float32) for _ in range(2)])
    aug_y = np.concatenate([train_labels] * 3)

    w0 = len(train_labels) / (2 * n0) if n0 > 0 else 1.0
    w1 = len(train_labels) / (2 * n1) if n1 > 0 else 1.0

    train_loader = DataLoader(TensorDataset(torch.FloatTensor(aug_X), torch.LongTensor(aug_y)), batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.FloatTensor(val_seqs), torch.LongTensor(val_labels)), batch_size=BATCH)
    test_loader = DataLoader(TensorDataset(torch.FloatTensor(test_seqs), torch.LongTensor(test_labels)), batch_size=BATCH)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feat_dim = len(feature_cols)
    model = CNNDirectionModel(feature_dim=feat_dim, sequence_length=SEQ_LEN, dropout=0.4).to(device)

    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([w0, w1]).to(device), label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)

    best_acc, p_cnt, best_epoch = 0, 0, 0
    low = crypto.lower()
    model_path = MODEL_DIR / f'{low}_short_cnn_model.pt'

    for epoch in range(EPOCHS):
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
        correct, total, ns, nn_ = 0, 0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).argmax(1)
                correct += (pred == yb).sum().item()
                total += yb.size(0)
                ns += (pred == 1).sum().item()
                nn_ += (pred == 0).sum().item()

        acc = correct / total if total > 0 else 0

        if acc > best_acc and ns >= 3 and nn_ >= 3:
            best_acc = acc
            best_epoch = epoch + 1
            p_cnt = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'feature_dim': feat_dim, 'sequence_length': SEQ_LEN,
                'model_type': 'cnn', 'direction': 'short',
                'short_tp_pct': SHORT_TP, 'short_sl_pct': SHORT_SL,
                'epoch': epoch + 1, 'val_acc': acc,
            }, model_path)
        else:
            p_cnt += 1

        if (epoch + 1) % 20 == 0:
            logger.info(f"  E{epoch+1}: acc={acc:.3f} S={ns} N={nn_} best={best_acc:.3f}")

        if p_cnt >= PATIENCE:
            break

    if not model_path.exists():
        logger.error(f"  {crypto}: No model saved (never found diverse predictions)")
        return None

    # Test on unseen data (2025 H2)
    ckpt = torch.load(model_path, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    all_dirs, all_confs, all_trues = [], [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            d, c = model.predict_direction(xb.to(device))
            all_dirs.extend(d.cpu().numpy())
            all_confs.extend(c.cpu().numpy())
            all_trues.extend(yb.numpy())

    ad, ac, at = np.array(all_dirs), np.array(all_confs), np.array(all_trues)

    logger.info(f"\n  {crypto} SHORT RESULTS | Best epoch: {best_epoch} | Val acc: {best_acc:.3f}")
    logger.info(f"  TEST (unseen 2025 H2):")
    for thresh in [0.50, 0.55, 0.60, 0.65, 0.70]:
        mask = (ad == 1) & (ac >= thresh)
        if mask.sum() > 0:
            wr = at[mask].mean() * 100
            logger.info(f"    Conf >= {thresh:.0%}: {mask.sum()} signals, WR: {wr:.1f}%")

    # Save scaler and features
    joblib.dump(scaler, MODEL_DIR / f'{low}_short_feature_scaler.joblib')
    with open(MODEL_DIR / f'{low}_short_features.json', 'w') as f:
        json.dump(feature_cols, f)

    return best_acc


COINS = {
    'BTC':  '2017-01-01',
    'ETH':  '2018-01-01',
    'SOL':  '2020-08-01',
    'DOGE': '2019-07-01',
    'AVAX': '2020-09-01',
}


def main():
    logger.info("=" * 70)
    logger.info("RETRAINING ALL SHORT MODELS (Walk-Forward)")
    logger.info("=" * 70)
    logger.info(f"SHORT TP={SHORT_TP:.1%} | SL={SHORT_SL:.1%} | SeqLen={SEQ_LEN}")

    results = {}
    for crypto, start in COINS.items():
        try:
            df, feat_cols = build_features(crypto, start)
            acc = train_short(crypto, df, feat_cols)
            results[crypto] = acc
        except Exception as e:
            logger.error(f"{crypto}: FAILED - {e}")
            results[crypto] = None

    logger.info("\n" + "=" * 70)
    logger.info("FINAL RESULTS")
    logger.info("=" * 70)
    for coin, acc in results.items():
        status = f"{acc:.3f}" if acc else "FAILED"
        logger.info(f"  {coin}: {status}")


if __name__ == "__main__":
    main()
