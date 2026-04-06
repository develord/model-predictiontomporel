"""
AVAX Production Retraining Script
==================================
Retrains AVAX models on the FULL dataset (no train/val split) for
production deployment:
  1. LONG CNN  (CNNDirectionModel)
  2. SHORT CNN (DeepCNNShortModel)

Key principles:
  - Train on ALL available data (no validation holdout)
  - Use FIXED number of epochs (from calibrated best_epoch)
  - Keep temperature values from calibrated runs (already optimized)
  - Save models in same locations as current ones
  - NO meta model (NoMeta = best for AVAX)

Usage:
    python 08_train_production.py
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
import ta as ta_lib

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR / 'scripts'))

from direction_prediction_model import CNNDirectionModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = BASE_DIR / 'data' / 'cache'
LONG_MODEL_DIR = BASE_DIR / 'models'
SHORT_MODEL_DIR = BASE_DIR / 'models_short'

LONG_MODEL_DIR.mkdir(parents=True, exist_ok=True)
SHORT_MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# DeepCNNShortModel (inline, same architecture as BTC SHORT)
# ============================================================
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
        self.conv9_1 = nn.Conv1d(96, 48, kernel_size=9, padding=4)
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
        return direction, confidence, probs


# ============================================================
# Shared params
# ============================================================
SEQUENCE_LENGTH = 30
GRAD_CLIP = 0.5

# ATR labeling (LONG)
ATR_TP_MULT = 1.5
ATR_SL_MULT = 1.5
FIXED_TP_PCT = 0.012
FIXED_SL_PCT = 0.012
BASE_LOOKAHEAD = 10
ATR_LOOKAHEAD_MULT = 0.7

# ATR labeling (SHORT) - symmetric
SHORT_ATR_TP_MULT = 1.5
SHORT_ATR_SL_MULT = 1.5
SHORT_FIXED_TP_PCT = 0.012
SHORT_FIXED_SL_PCT = 0.012
SHORT_BASE_LOOKAHEAD = 10
SHORT_ATR_LOOKAHEAD_MULT = 0.7

# ============================================================
# LONG CNN params
# ============================================================
LONG_FIXED_EPOCHS = 18     # Best epoch from calibrated run (stopped 53, best 18)
LONG_BATCH_SIZE = 64
LONG_LR = 0.0015
LONG_NOISE_STD = 0.015
LONG_LABEL_SMOOTHING = 0.1
LONG_AUGMENT_COPIES = 2    # 3x total (original + 2 copies)
LONG_TEMPERATURE = 2.062   # From AVAX calibrated run

# ============================================================
# SHORT CNN params
# ============================================================
SHORT_FIXED_EPOCHS = 14    # Best epoch from calibrated run (stopped 54, best 14)
SHORT_BATCH_SIZE = 64
SHORT_LR = 0.0005
SHORT_NOISE_STD = 0.01
SHORT_LABEL_SMOOTHING = 0.05
SHORT_AUGMENT_COPIES = 3   # 4x total
SHORT_TEMPERATURE = 1.495  # From AVAX calibrated run


# ============================================================
# Label creation
# ============================================================
def create_long_labels(df):
    """ATR-adaptive LONG labels."""
    atr = ta_lib.volatility.AverageTrueRange(
        df['high'], df['low'], df['close'], window=14
    ).average_true_range()
    median_atr = atr.rolling(50).median()

    labels = []
    for i in range(len(df)):
        if i >= len(df) - 1:
            labels.append(-1)
            continue
        entry = df.iloc[i]['close']
        cur_atr = atr.iloc[i]
        if pd.notna(cur_atr) and cur_atr > 0:
            tp_dist = cur_atr * ATR_TP_MULT
            sl_dist = cur_atr * ATR_SL_MULT
        else:
            tp_dist = entry * FIXED_TP_PCT
            sl_dist = entry * FIXED_SL_PCT
        tp = entry + tp_dist
        sl = entry - sl_dist
        med = median_atr.iloc[i] if pd.notna(median_atr.iloc[i]) and median_atr.iloc[i] > 0 else cur_atr
        if pd.notna(med) and med > 0:
            vol_ratio = cur_atr / med
            lookahead = int(BASE_LOOKAHEAD * max(0.5, min(2.0, vol_ratio * ATR_LOOKAHEAD_MULT + 0.3)))
        else:
            lookahead = BASE_LOOKAHEAD
        lookahead = max(5, min(20, lookahead))
        hit_tp, hit_sl = False, False
        for j in range(i + 1, min(i + 1 + lookahead, len(df))):
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


def create_short_labels(df):
    """ATR-adaptive SHORT labels (symmetric TP/SL)."""
    atr = ta_lib.volatility.AverageTrueRange(
        df['high'], df['low'], df['close'], window=14
    ).average_true_range()
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
        tp = entry - tp_dist  # SHORT TP = price drops
        sl = entry + sl_dist  # SHORT SL = price rises
        med = median_atr.iloc[i] if pd.notna(median_atr.iloc[i]) and median_atr.iloc[i] > 0 else cur_atr
        if pd.notna(med) and med > 0:
            vol_ratio = cur_atr / med
            lookahead = int(SHORT_BASE_LOOKAHEAD * max(0.5, min(2.0, vol_ratio * SHORT_ATR_LOOKAHEAD_MULT + 0.3)))
        else:
            lookahead = SHORT_BASE_LOOKAHEAD
        lookahead = max(5, min(20, lookahead))
        hit_tp, hit_sl = False, False
        for j in range(i + 1, min(i + 1 + lookahead, len(df))):
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


# ============================================================
# Bear features (for SHORT model)
# ============================================================
def add_bear_features(df):
    """Add features specifically useful for detecting SHORT opportunities."""
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
        np.where((df['price_change_5'] < 0) & (df['vol_change_5'] > 0), -1, 0)
    )

    df['is_red'] = (df['close'] < df['open']).astype(int)
    df['consec_red'] = 0
    for i in range(1, len(df)):
        if df.iloc[i]['is_red']:
            df.iloc[i, df.columns.get_loc('consec_red')] = df.iloc[i - 1]['consec_red'] + 1

    df['high_rejection'] = (df['high'] - df['close']) / (df['high'] - df['low'] + 1e-10)
    df['dist_from_high_20'] = (df['close'] / df['high'].rolling(20).max() - 1) * 100
    df['dist_from_high_50'] = (df['close'] / df['high'].rolling(50).max() - 1) * 100
    df['vol_expansion'] = df['1d_atr_14'] / df['1d_atr_14'].shift(5) if '1d_atr_14' in df.columns else 1

    if '1d_rsi_14' in df.columns:
        df['rsi_slope_5'] = df['1d_rsi_14'].diff(5)
        df['price_slope_5'] = df['close'].pct_change(5) * 100
        df['rsi_price_divergence'] = np.where(
            (df['price_slope_5'] > 0) & (df['rsi_slope_5'] < 0), 1,
            np.where((df['price_slope_5'] < 0) & (df['rsi_slope_5'] > 0), -1, 0)
        )
    return df


# ============================================================
# Training helpers
# ============================================================
def train_epoch_weighted(model, dataloader, criterion, optimizer, device, grad_clip):
    """Train one epoch with sample weights."""
    model.train()
    total_loss, correct, total = 0, 0, 0
    for batch in dataloader:
        if len(batch) == 3:
            X_b, y_b, w_b = batch
            X_b, y_b, w_b = X_b.to(device), y_b.to(device), w_b.to(device)
        else:
            X_b, y_b = batch[0].to(device), batch[1].to(device)
            w_b = None
        optimizer.zero_grad()
        logits = model(X_b)
        loss = criterion(logits, y_b)
        if w_b is not None:
            loss = (loss * w_b).mean() if loss.dim() > 0 else loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()
        _, pred = torch.max(logits, 1)
        correct += (pred == y_b).sum().item()
        total += y_b.size(0)
    return total_loss / len(dataloader), correct / total


def train_epoch_simple(model, dataloader, criterion, optimizer, device, grad_clip):
    """Train one epoch without sample weights."""
    model.train()
    total_loss, correct, total = 0, 0, 0
    for X_b, y_b in dataloader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_b), y_b)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()
        _, pred = torch.max(model(X_b), 1)
        correct += (pred == y_b).sum().item()
        total += y_b.size(0)
    return total_loss / len(dataloader), correct / total


# ============================================================
# STEP 1: Train LONG CNN on full data
# ============================================================
def train_long_cnn():
    logger.info(f"\n{'='*70}")
    logger.info(f"STEP 1: LONG CNN - Production Retrain (ALL data, {LONG_FIXED_EPOCHS} epochs)")
    logger.info(f"{'='*70}\n")

    df = pd.read_csv(DATA_DIR / 'avax_features.csv')
    df['date'] = pd.to_datetime(df['date'])
    logger.info(f"Data: {len(df)} rows, {df['date'].min().date()} to {df['date'].max().date()}")

    with open(BASE_DIR / 'required_features.json') as f:
        feature_cols = json.load(f)
    FEATURE_DIM = len(feature_cols)
    logger.info(f"Features: {FEATURE_DIM}")

    # Labels
    df['label'] = create_long_labels(df)
    n_tp = (df['label'] == 1).sum()
    n_sl = (df['label'] == 0).sum()
    n_unk = (df['label'] == -1).sum()
    logger.info(f"Labels: TP={n_tp}, SL={n_sl}, unknown={n_unk}")

    # Scale ALL data
    X_raw = df[feature_cols].fillna(0).values.astype(np.float32)
    scaler = RobustScaler()
    X_scaled = np.clip(
        np.nan_to_num(scaler.fit_transform(X_raw), nan=0.0, posinf=0.0, neginf=0.0),
        -5, 5
    ).astype(np.float32)
    joblib.dump(scaler, LONG_MODEL_DIR / 'feature_scaler.joblib')
    logger.info(f"Scaler saved: {LONG_MODEL_DIR / 'feature_scaler.joblib'}")

    # Build sequences with bear weighting on ALL data
    y_raw = df['label'].values
    bear_flag = df['is_strong_bear'].values if 'is_strong_bear' in df.columns else np.zeros(len(df))

    train_seqs, train_labels, train_weights = [], [], []
    for i in range(SEQUENCE_LENGTH, len(X_scaled)):
        if y_raw[i] != -1:
            train_seqs.append(X_scaled[i - SEQUENCE_LENGTH:i])
            train_labels.append(y_raw[i])
            is_bear = bear_flag[i] if i < len(bear_flag) else 0
            if is_bear and y_raw[i] == 1:
                train_weights.append(0.3)
            elif is_bear and y_raw[i] == 0:
                train_weights.append(2.0)
            else:
                train_weights.append(1.0)

    train_seqs = np.array(train_seqs)
    train_labels = np.array(train_labels)
    train_weights = np.array(train_weights, dtype=np.float32)

    n_sl = (train_labels == 0).sum()
    n_tp = (train_labels == 1).sum()
    n_bear_tp = ((train_labels == 1) & (train_weights < 1.0)).sum()
    n_bear_sl = ((train_labels == 0) & (train_weights > 1.0)).sum()
    logger.info(f"Sequences: {len(train_seqs)} | SL={n_sl} TP={n_tp}")
    logger.info(f"Bear-weighted: {n_bear_tp} TP downweighted (0.3x), {n_bear_sl} SL upweighted (2x)")

    # Augmentation (3x total)
    aug_seqs, aug_labels, aug_weights = [train_seqs], [train_labels], [train_weights]
    for _ in range(LONG_AUGMENT_COPIES):
        noise = np.random.normal(0, LONG_NOISE_STD, train_seqs.shape).astype(np.float32)
        aug_seqs.append(train_seqs + noise)
        aug_labels.append(train_labels)
        aug_weights.append(train_weights)
    train_seqs_aug = np.concatenate(aug_seqs)
    train_labels_aug = np.concatenate(aug_labels)
    train_weights_aug = np.concatenate(aug_weights)
    logger.info(f"After augmentation: {len(train_seqs_aug)} sequences (3x)")

    # Class weights
    weight_sl = len(train_labels) / (2 * n_sl) if n_sl > 0 else 1.0
    weight_tp = len(train_labels) / (2 * n_tp) if n_tp > 0 else 1.0
    class_weights = torch.FloatTensor([weight_sl, weight_tp])

    # Dataloader
    train_dataset = TensorDataset(
        torch.FloatTensor(train_seqs_aug),
        torch.LongTensor(train_labels_aug),
        torch.FloatTensor(train_weights_aug)
    )
    train_loader = DataLoader(train_dataset, batch_size=LONG_BATCH_SIZE, shuffle=True)

    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNNDirectionModel(feature_dim=FEATURE_DIM, sequence_length=SEQUENCE_LENGTH, dropout=0.4).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: CNNDirectionModel | Params: {n_params:,} | Device: {device}")

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=LONG_LABEL_SMOOTHING,
        reduction='none'
    )
    optimizer = optim.AdamW(model.parameters(), lr=LONG_LR, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)

    logger.info(f"Training {LONG_FIXED_EPOCHS} fixed epochs (no early stopping)...")
    for epoch in range(LONG_FIXED_EPOCHS):
        train_loss, train_acc = train_epoch_weighted(
            model, train_loader, criterion, optimizer, device, GRAD_CLIP
        )
        scheduler.step(epoch)
        logger.info(f"  E{epoch+1:2d}/{LONG_FIXED_EPOCHS} | Loss: {train_loss:.4f} | Acc: {train_acc:.3f} | "
                     f"LR: {optimizer.param_groups[0]['lr']:.6f}")

    # Save with hardcoded temperature
    torch.save({
        'model_state_dict': model.state_dict(),
        'feature_dim': FEATURE_DIM,
        'sequence_length': SEQUENCE_LENGTH,
        'model_type': 'cnn',
        'temperature': LONG_TEMPERATURE,
        'production_retrain': True,
        'epochs_trained': LONG_FIXED_EPOCHS,
    }, LONG_MODEL_DIR / 'AVAX_direction_model.pt')

    size_kb = (LONG_MODEL_DIR / 'AVAX_direction_model.pt').stat().st_size / 1024
    logger.info(f"Saved: {LONG_MODEL_DIR / 'AVAX_direction_model.pt'} ({size_kb:.0f} KB)")
    logger.info(f"Temperature (hardcoded): {LONG_TEMPERATURE}")

    return model, FEATURE_DIM


# ============================================================
# STEP 2: Train SHORT CNN on full data (DeepCNNShortModel)
# ============================================================
def train_short_cnn():
    logger.info(f"\n{'='*70}")
    logger.info(f"STEP 2: SHORT CNN - Production Retrain (ALL data, {SHORT_FIXED_EPOCHS} epochs)")
    logger.info(f"{'='*70}\n")

    df = pd.read_csv(DATA_DIR / 'avax_features.csv')
    df['date'] = pd.to_datetime(df['date'])
    logger.info(f"Data: {len(df)} rows")

    # Load base features
    with open(BASE_DIR / 'required_features.json') as f:
        base_features = json.load(f)

    # Add bear features
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
    logger.info(f"Creating SHORT labels (ATR TP={SHORT_ATR_TP_MULT}x, SL={SHORT_ATR_SL_MULT}x)...")
    df['label'] = create_short_labels(df)
    n1 = (df['label'] == 1).sum()
    n0 = (df['label'] == 0).sum()
    logger.info(f"  SHORT profitable: {n1} | No short: {n0} | Ratio: {n1/(n0+1):.2f}")

    # Scale ALL data
    X_raw = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    scaler = RobustScaler()
    X_scaled = np.clip(
        np.nan_to_num(scaler.fit_transform(X_raw), nan=0, posinf=0, neginf=0),
        -5, 5
    ).astype(np.float32)
    joblib.dump(scaler, SHORT_MODEL_DIR / 'feature_scaler_short.joblib')

    # Build sequences on ALL data
    y_raw = df['label'].values
    seqs, labels = [], []
    for i in range(SEQUENCE_LENGTH, len(X_scaled)):
        if y_raw[i] != -1:
            seqs.append(X_scaled[i - SEQUENCE_LENGTH:i])
            labels.append(y_raw[i])
    train_seqs = np.array(seqs)
    train_labels = np.array(labels)

    n_0 = (train_labels == 0).sum()
    n_1 = (train_labels == 1).sum()
    logger.info(f"Sequences: {len(train_seqs)} (No={n_0}, Short={n_1})")

    # Augment (4x total)
    aug_seqs, aug_labels = [train_seqs], [train_labels]
    for _ in range(SHORT_AUGMENT_COPIES):
        aug_seqs.append(train_seqs + np.random.normal(0, SHORT_NOISE_STD, train_seqs.shape).astype(np.float32))
        aug_labels.append(train_labels)
    train_seqs_aug = np.concatenate(aug_seqs)
    train_labels_aug = np.concatenate(aug_labels)
    logger.info(f"After augmentation: {len(train_seqs_aug)} (4x)")

    # Class weights
    w0 = len(train_labels) / (2 * n_0) if n_0 > 0 else 1.0
    w1 = len(train_labels) / (2 * n_1) if n_1 > 0 else 1.0

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(train_seqs_aug), torch.LongTensor(train_labels_aug)),
        batch_size=SHORT_BATCH_SIZE, shuffle=True
    )

    # Model - DeepCNNShortModel for AVAX SHORT
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepCNNShortModel(feature_dim=FEATURE_DIM, sequence_length=SEQUENCE_LENGTH, dropout=0.35).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: DeepCNNShortModel (SHORT) | Params: {n_params:,} | Device: {device}")

    criterion = nn.CrossEntropyLoss(
        weight=torch.FloatTensor([w0, w1]).to(device),
        label_smoothing=SHORT_LABEL_SMOOTHING
    )
    optimizer = optim.AdamW(model.parameters(), lr=SHORT_LR, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

    logger.info(f"Training {SHORT_FIXED_EPOCHS} fixed epochs (no early stopping)...")
    for epoch in range(SHORT_FIXED_EPOCHS):
        train_loss, train_acc = train_epoch_simple(
            model, train_loader, criterion, optimizer, device, GRAD_CLIP
        )
        scheduler.step(epoch)
        logger.info(f"  E{epoch+1:2d}/{SHORT_FIXED_EPOCHS} | Loss: {train_loss:.4f} | Acc: {train_acc:.3f} | "
                     f"LR: {optimizer.param_groups[0]['lr']:.6f}")

    # Save with hardcoded temperature
    torch.save({
        'model_state_dict': model.state_dict(),
        'feature_dim': FEATURE_DIM,
        'sequence_length': SEQUENCE_LENGTH,
        'model_type': 'deep_cnn_short',
        'temperature': SHORT_TEMPERATURE,
        'short_atr_tp_mult': SHORT_ATR_TP_MULT,
        'short_atr_sl_mult': SHORT_ATR_SL_MULT,
        'production_retrain': True,
        'epochs_trained': SHORT_FIXED_EPOCHS,
    }, SHORT_MODEL_DIR / 'AVAX_short_model.pt')

    # Save feature list
    with open(SHORT_MODEL_DIR / 'short_features.json', 'w') as f:
        json.dump(feature_cols, f)

    size_kb = (SHORT_MODEL_DIR / 'AVAX_short_model.pt').stat().st_size / 1024
    logger.info(f"Saved: {SHORT_MODEL_DIR / 'AVAX_short_model.pt'} ({size_kb:.0f} KB)")
    logger.info(f"Temperature (hardcoded): {SHORT_TEMPERATURE}")
    logger.info(f"Features saved: {SHORT_MODEL_DIR / 'short_features.json'}")

    return model, FEATURE_DIM, feature_cols


# ============================================================
# MAIN
# ============================================================
def main():
    logger.info(f"\n{'#'*70}")
    logger.info(f"#  AVAX PRODUCTION RETRAIN - CNN MODELS ON FULL DATASET (NO META)")
    logger.info(f"#  No train/val split | Fixed epochs | Hardcoded temperatures")
    logger.info(f"{'#'*70}\n")

    # Step 1: LONG CNN
    long_model, long_dim = train_long_cnn()

    # Step 2: SHORT CNN
    short_model, short_dim, short_features = train_short_cnn()

    # No Step 3: Meta XGBoost skipped (NoMeta = best for AVAX)

    logger.info(f"\n{'#'*70}")
    logger.info(f"#  PRODUCTION RETRAIN COMPLETE")
    logger.info(f"{'#'*70}")
    logger.info(f"  LONG CNN:       {LONG_MODEL_DIR / 'AVAX_direction_model.pt'}")
    logger.info(f"  LONG scaler:    {LONG_MODEL_DIR / 'feature_scaler.joblib'}")
    logger.info(f"  SHORT CNN:      {SHORT_MODEL_DIR / 'AVAX_short_model.pt'}")
    logger.info(f"  SHORT scaler:   {SHORT_MODEL_DIR / 'feature_scaler_short.joblib'}")
    logger.info(f"  SHORT features: {SHORT_MODEL_DIR / 'short_features.json'}")
    logger.info(f"  Meta:           SKIPPED (NoMeta = best for AVAX)")
    logger.info(f"\n  Temperatures: LONG={LONG_TEMPERATURE}, SHORT={SHORT_TEMPERATURE}")
    logger.info(f"  Epochs: LONG={LONG_FIXED_EPOCHS}, SHORT={SHORT_FIXED_EPOCHS}")


if __name__ == "__main__":
    main()
