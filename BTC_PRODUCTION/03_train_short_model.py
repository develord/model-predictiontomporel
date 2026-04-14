"""
BTC SHORT CNN Training Script V2
==================================
Improved SHORT model with:
- Better TP/SL params (2% TP, 1% SL) for balanced labels
- Bear-specific features added to existing features
- Deeper CNN architecture
- Longer sequence (45 days) to capture distribution patterns
- More aggressive augmentation

Usage:
    python 03_train_short_model.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import logging
import json
import joblib
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report
from typing import Tuple

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR / 'scripts'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = BASE_DIR / 'data' / 'cache'
MODEL_DIR = BASE_DIR / 'models_short'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SEQUENCE_LENGTH = 30  # Match LONG model
BATCH_SIZE = 64
EPOCHS = 250
LEARNING_RATE = 0.0005
PATIENCE = 40
GRAD_CLIP = 0.5
NOISE_STD = 0.01
LABEL_SMOOTHING = 0.1

# ATR-based SHORT labeling (symmetric = no bias)
SHORT_ATR_TP_MULT = 1.5   # TP = 1.5x ATR drop
SHORT_ATR_SL_MULT = 1.5   # SL = 1.5x ATR rise (symmetric)
SHORT_FIXED_TP_PCT = 0.012
SHORT_FIXED_SL_PCT = 0.012
SHORT_BASE_LOOKAHEAD = 10
SHORT_ATR_LOOKAHEAD_MULT = 0.7

TRAIN_END = '2025-06-30'
VAL_START = '2025-07-01'
VAL_END = '2025-12-31'


class DeepCNNShortModel(nn.Module):
    """Deeper CNN specifically designed for SHORT detection.
    More conv layers to detect complex distribution/top patterns."""

    def __init__(self, feature_dim, sequence_length=45, dropout=0.35):
        super().__init__()

        # Feature projection
        self.input_proj = nn.Sequential(
            nn.Linear(feature_dim, 96),
            nn.BatchNorm1d(sequence_length),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Multi-scale conv block 1
        self.conv3_1 = nn.Conv1d(96, 48, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv1d(96, 48, kernel_size=5, padding=2)
        self.conv9_1 = nn.Conv1d(96, 48, kernel_size=9, padding=4)  # Wider for longer patterns
        self.bn1 = nn.BatchNorm1d(144)
        self.drop1 = nn.Dropout(dropout)

        # Conv block 2
        self.conv2 = nn.Sequential(
            nn.Conv1d(144, 96, kernel_size=3, padding=1),
            nn.BatchNorm1d(96),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Conv block 3 (deeper)
        self.conv3 = nn.Sequential(
            nn.Conv1d(96, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout * 0.7)
        )

        # Temporal attention
        self.attention = nn.Sequential(
            nn.Linear(64, 24),
            nn.Tanh(),
            nn.Linear(24, 1)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(64, 48),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(48, 24),
            nn.GELU(),
            nn.Linear(24, 2)
        )

    def forward(self, x):
        x = self.input_proj(x)  # (B, T, 96)
        x = x.permute(0, 2, 1)  # (B, 96, T)

        c3 = F.gelu(self.conv3_1(x))
        c5 = F.gelu(self.conv5_1(x))
        c9 = F.gelu(self.conv9_1(x))
        x = torch.cat([c3, c5, c9], dim=1)  # (B, 144, T)
        x = self.bn1(x)
        x = self.drop1(x)

        x = self.conv2(x)  # (B, 96, T)
        x = self.conv3(x)  # (B, 64, T)

        x = x.permute(0, 2, 1)  # (B, T, 64)
        attn = F.softmax(self.attention(x), dim=1)
        x = (x * attn).sum(dim=1)  # (B, 64)

        return self.classifier(x)

    def predict_direction(self, x):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
            confidence, direction = torch.max(probs, dim=1)
        return direction, confidence


def add_bear_features(df):
    """Add features specifically useful for detecting SHORT opportunities"""
    # Price vs moving averages (overbought = good short setup)
    for w in [10, 20, 50]:
        sma = df['close'].rolling(w).mean()
        df[f'price_above_sma{w}_pct'] = (df['close'] / sma - 1) * 100

    # Rate of change acceleration (slowing uptrend = distribution)
    df['roc_5'] = df['close'].pct_change(5) * 100
    df['roc_10'] = df['close'].pct_change(10) * 100
    df['roc_deceleration'] = df['roc_5'] - df['roc_10']  # Negative = slowing

    # Volume divergence (price up + volume down = weak rally)
    df['price_change_5'] = df['close'].pct_change(5)
    df['vol_change_5'] = df['volume'].pct_change(5)
    df['vol_price_divergence'] = np.where(
        (df['price_change_5'] > 0) & (df['vol_change_5'] < 0), 1,
        np.where((df['price_change_5'] < 0) & (df['vol_change_5'] > 0), -1, 0)
    )

    # Consecutive red candles
    df['is_red'] = (df['close'] < df['open']).astype(int)
    df['consec_red'] = 0
    for i in range(1, len(df)):
        if df.iloc[i]['is_red']:
            df.iloc[i, df.columns.get_loc('consec_red')] = df.iloc[i-1]['consec_red'] + 1

    # High-to-close ratio (rejection from highs = bearish)
    df['high_rejection'] = (df['high'] - df['close']) / (df['high'] - df['low'] + 1e-10)

    # Distance from recent high (how far from local top)
    df['dist_from_high_20'] = (df['close'] / df['high'].rolling(20).max() - 1) * 100
    df['dist_from_high_50'] = (df['close'] / df['high'].rolling(50).max() - 1) * 100

    # Volatility expansion (sudden vol increase = potential reversal)
    df['vol_expansion'] = df['1d_atr_14'] / df['1d_atr_14'].shift(5) if '1d_atr_14' in df.columns else 1

    # RSI divergence (price higher high but RSI lower high)
    if '1d_rsi_14' in df.columns:
        df['rsi_slope_5'] = df['1d_rsi_14'].diff(5)
        df['price_slope_5'] = df['close'].pct_change(5) * 100
        df['rsi_price_divergence'] = np.where(
            (df['price_slope_5'] > 0) & (df['rsi_slope_5'] < 0), 1,  # Bearish divergence
            np.where((df['price_slope_5'] < 0) & (df['rsi_slope_5'] > 0), -1, 0)  # Bullish div
        )

    return df


def create_short_labels(df):
    """ATR-adaptive SHORT labels (symmetric TP/SL)"""
    import ta as ta_lib
    atr = ta_lib.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
    median_atr = atr.rolling(50).median()
    labels = []
    for i in range(len(df)):
        if i >= len(df) - 1:
            labels.append(-1)
            continue
        entry = df.iloc[i]['close']
        cur_atr = atr.iloc[i]
        if pd.notna(cur_atr) and cur_atr > 0:
            tp_dist = cur_atr * SHORT_ATR_TP_MULT
            sl_dist = cur_atr * SHORT_ATR_SL_MULT
        else:
            tp_dist = entry * SHORT_FIXED_TP_PCT
            sl_dist = entry * SHORT_FIXED_SL_PCT
        tp = entry - tp_dist   # SHORT TP = price drops
        sl = entry + sl_dist   # SHORT SL = price rises
        med = median_atr.iloc[i] if pd.notna(median_atr.iloc[i]) and median_atr.iloc[i] > 0 else cur_atr
        if pd.notna(med) and med > 0:
            vol_ratio = cur_atr / med
            lookahead = int(SHORT_BASE_LOOKAHEAD * max(0.5, min(2.0, vol_ratio * SHORT_ATR_LOOKAHEAD_MULT + 0.3)))
        else:
            lookahead = SHORT_BASE_LOOKAHEAD
        lookahead = max(5, min(20, lookahead))
        hit_tp, hit_sl = False, False
        for j in range(i+1, min(i+1+lookahead, len(df))):
            if df.iloc[j]['low'] <= tp:
                hit_tp = True
                break
            if df.iloc[j]['high'] >= sl:
                hit_sl = True
                break
        if hit_tp:
            labels.append(1)
        elif hit_sl:
            labels.append(0)
        else:
            labels.append(-1)
    return np.array(labels)


def build_sequences(X, y, seq_len):
    seqs, labels = [], []
    for i in range(seq_len, len(X)):
        if y[i] != -1:
            seqs.append(X[i-seq_len:i])
            labels.append(y[i])
    return np.array(seqs), np.array(labels)


def augment(X, y, noise=0.01, copies=3):
    ax, ay = [X], [y]
    for _ in range(copies):
        ax.append(X + np.random.normal(0, noise, X.shape).astype(np.float32))
        ay.append(y)
    return np.concatenate(ax), np.concatenate(ay)


def train():
    logger.info(f"\n{'='*70}")
    logger.info(f"BTC SHORT CNN V2 TRAINING (DeepCNN + Bear Features)")
    logger.info(f"{'='*70}\n")

    df = pd.read_csv(DATA_DIR / 'btc_features.csv')
    df['date'] = pd.to_datetime(df['date'])
    logger.info(f"Data: {len(df)} rows")

    # Load base features
    with open(BASE_DIR / 'required_features.json') as f:
        base_features = json.load(f)

    # Add bear-specific features
    logger.info("Adding bear-specific features...")
    df = add_bear_features(df)

    bear_features = [
        'price_above_sma10_pct', 'price_above_sma20_pct', 'price_above_sma50_pct',
        'roc_5', 'roc_10', 'roc_deceleration',
        'vol_price_divergence', 'consec_red', 'high_rejection',
        'dist_from_high_20', 'dist_from_high_50', 'vol_expansion',
    ]
    if 'rsi_price_divergence' in df.columns:
        bear_features.extend(['rsi_slope_5', 'price_slope_5', 'rsi_price_divergence'])

    feature_cols = base_features + [f for f in bear_features if f in df.columns]
    FEATURE_DIM = len(feature_cols)
    logger.info(f"Features: {FEATURE_DIM} ({len(base_features)} base + {FEATURE_DIM - len(base_features)} bear)")

    # Labels
    logger.info(f"Creating SHORT labels (ATR-based, TP={SHORT_ATR_TP_MULT}x ATR, SL={SHORT_ATR_SL_MULT}x ATR)...")
    df['label'] = create_short_labels(df)

    n1 = (df['label'] == 1).sum()
    n0 = (df['label'] == 0).sum()
    logger.info(f"  SHORT profitable: {n1} | No short: {n0} | Ratio: {n1/(n0+1):.2f}")

    # Split
    train_df = df[df['date'] <= TRAIN_END].copy()
    val_df = df[(df['date'] >= VAL_START) & (df['date'] <= VAL_END)].copy()

    # Scale
    X_train_raw = train_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    X_val_raw = val_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    y_train_raw = train_df['label'].values
    y_val_raw = val_df['label'].values

    scaler = RobustScaler()
    X_train = np.clip(np.nan_to_num(scaler.fit_transform(X_train_raw), nan=0, posinf=0, neginf=0), -5, 5).astype(np.float32)
    X_val = np.clip(np.nan_to_num(scaler.transform(X_val_raw), nan=0, posinf=0, neginf=0), -5, 5).astype(np.float32)

    joblib.dump(scaler, MODEL_DIR / 'feature_scaler_short.joblib')

    # Sequences
    train_seqs, train_labels = build_sequences(X_train, y_train_raw, SEQUENCE_LENGTH)
    val_seqs, val_labels = build_sequences(X_val, y_val_raw, SEQUENCE_LENGTH)

    n_0, n_1 = (train_labels == 0).sum(), (train_labels == 1).sum()
    logger.info(f"Train: {len(train_seqs)} (No={n_0}, Short={n_1}) | Val: {len(val_seqs)}")

    # Augment (3x)
    train_seqs_a, train_labels_a = augment(train_seqs, train_labels, NOISE_STD, copies=3)
    logger.info(f"After augmentation: {len(train_seqs_a)} (4x)")

    # Class weights
    w0 = len(train_labels) / (2 * n_0) if n_0 > 0 else 1.0
    w1 = len(train_labels) / (2 * n_1) if n_1 > 0 else 1.0

    train_loader = DataLoader(TensorDataset(torch.FloatTensor(train_seqs_a), torch.LongTensor(train_labels_a)),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.FloatTensor(val_seqs), torch.LongTensor(val_labels)),
                            batch_size=BATCH_SIZE)

    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from direction_prediction_model import CNNDirectionModel
    model = CNNDirectionModel(feature_dim=FEATURE_DIM, sequence_length=SEQUENCE_LENGTH, dropout=0.4).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: CNNDirectionModel (SHORT) | Params: {n_params:,} | Device: {device}")

    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([w0, w1]).to(device), label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

    best_acc, patience_cnt, best_epoch = 0, 0, 0

    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
        scheduler.step(epoch)

        model.eval()
        correct, total, n_short, n_no = 0, 0, 0, 0
        all_p, all_l = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).argmax(1)
                correct += (pred == yb).sum().item()
                total += yb.size(0)
                n_short += (pred == 1).sum().item()
                n_no += (pred == 0).sum().item()
                all_p.extend(pred.cpu().numpy())
                all_l.extend(yb.cpu().numpy())

        acc = correct / total if total > 0 else 0
        pa, la = np.array(all_p), np.array(all_l)
        sp = la[pa == 1].mean() * 100 if n_short > 0 else 0

        if (epoch + 1) % 10 == 0:
            logger.info(f"E{epoch+1:3d}/{EPOCHS} | Acc:{acc:.3f} | SHORT={n_short} NO={n_no} | ShortPrec={sp:.0f}%")

        if acc > best_acc and n_short >= 5 and n_no >= 5:
            best_acc = acc
            best_epoch = epoch + 1
            patience_cnt = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'feature_dim': FEATURE_DIM,
                'sequence_length': SEQUENCE_LENGTH,
                'model_type': 'cnn',
                'short_atr_tp_mult': SHORT_ATR_TP_MULT,
                'short_atr_sl_mult': SHORT_ATR_SL_MULT,
                'epoch': epoch + 1,
                'val_acc': acc,
            }, MODEL_DIR / 'BTC_short_model.pt')
        else:
            patience_cnt += 1

        if patience_cnt >= PATIENCE:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    # Final eval
    ckpt = torch.load(MODEL_DIR / 'BTC_short_model.pt')
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    all_confs, all_dirs, all_trues = [], [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            d, c = model.predict_direction(xb.to(device))
            all_dirs.extend(d.cpu().numpy())
            all_confs.extend(c.cpu().numpy())
            all_trues.extend(yb.numpy())

    logger.info(f"\n{'='*70}")
    logger.info(f"TRAINING COMPLETE | Best epoch: {best_epoch} | Best acc: {best_acc:.3f}")
    logger.info(f"{'='*70}")

    # === TEMPERATURE SCALING (Confidence Calibration) ===
    logger.info(f"\nTemperature Scaling (Confidence Calibration)...")
    all_logits_t, all_labels_t = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            logits = model(xb.to(device))
            all_logits_t.append(logits.cpu())
            all_labels_t.append(yb)
    all_logits_t = torch.cat(all_logits_t)
    all_labels_t = torch.cat(all_labels_t)

    temperature = nn.Parameter(torch.ones(1) * 1.5)
    temp_optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=100)
    nll_criterion = nn.CrossEntropyLoss()

    def temp_eval():
        temp_optimizer.zero_grad()
        loss = nll_criterion(all_logits_t / temperature, all_labels_t)
        loss.backward()
        return loss

    temp_optimizer.step(temp_eval)
    optimal_temp = temperature.item()
    logger.info(f"  Optimal temperature: {optimal_temp:.4f}")

    # Save temperature in checkpoint
    ckpt_data = torch.load(MODEL_DIR / 'BTC_short_model.pt')
    ckpt_data['temperature'] = optimal_temp
    torch.save(ckpt_data, MODEL_DIR / 'BTC_short_model.pt')

    # Calibrated confidence analysis
    all_confs, all_dirs, all_trues = [], [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            logits = model(xb.to(device))
            calibrated_probs = torch.softmax(logits / optimal_temp, dim=1)
            confidence, direction = torch.max(calibrated_probs, dim=1)
            all_dirs.extend(direction.cpu().numpy())
            all_confs.extend(confidence.cpu().numpy())
            all_trues.extend(yb.numpy())

    ad, ac, at = np.array(all_dirs), np.array(all_confs), np.array(all_trues)

    logger.info(f"\nCalibrated Confidence Analysis (SHORT signals):")
    for thresh in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        mask = (ad == 1) & (ac >= thresh)
        if mask.sum() > 0:
            wr = at[mask].mean() * 100
            logger.info(f"  Conf >= {thresh:.0%}: {mask.sum()} signals, WR: {wr:.1f}%")

    with open(MODEL_DIR / 'short_features.json', 'w') as f:
        json.dump(feature_cols, f)

    logger.info(f"\nModel: {(MODEL_DIR / 'BTC_short_model.pt').stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    train()
