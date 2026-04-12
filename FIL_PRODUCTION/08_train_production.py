"""
FIL Production Retraining Script
==================================
Retrains FIL models on the FULL dataset (no train/val split) for
production deployment:
  1. LONG CNN  (CNNDirectionModel)
  2. SHORT CNN (DeepCNNShortModel)
  3. META XGBoost (LONG + SHORT)

Key principles:
  - Train on ALL available data (no validation holdout)
  - Use FIXED number of epochs (from calibrated best_epoch)
  - Keep temperature values from calibrated runs (already optimized)
  - Save models in same locations as current ones
  - Meta models retrained on full CNN predictions

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


# ============================================================
# DeepCNNShortModel (inline, same architecture as training script)
# ============================================================
class DeepCNNShortModel(nn.Module):
    """Deeper CNN specifically designed for SHORT detection."""

    def __init__(self, feature_dim, sequence_length=45, dropout=0.35):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(feature_dim, 96),
            nn.BatchNorm1d(sequence_length),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.conv3_1 = nn.Conv1d(96, 48, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv1d(96, 48, kernel_size=5, padding=2)
        self.conv9_1 = nn.Conv1d(96, 48, kernel_size=9, padding=4)
        self.bn1 = nn.BatchNorm1d(144)
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Sequential(
            nn.Conv1d(144, 96, kernel_size=3, padding=1),
            nn.BatchNorm1d(96),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(96, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout * 0.7)
        )

        self.attention = nn.Sequential(
            nn.Linear(64, 24),
            nn.Tanh(),
            nn.Linear(24, 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(64, 48),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(48, 24),
            nn.GELU(),
            nn.Linear(24, 2)
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = x.permute(0, 2, 1)

        c3 = F.gelu(self.conv3_1(x))
        c5 = F.gelu(self.conv5_1(x))
        c9 = F.gelu(self.conv9_1(x))
        x = torch.cat([c3, c5, c9], dim=1)
        x = self.bn1(x)
        x = self.drop1(x)

        x = self.conv2(x)
        x = self.conv3(x)

        x = x.permute(0, 2, 1)
        attn = F.softmax(self.attention(x), dim=1)
        x = (x * attn).sum(dim=1)

        return self.classifier(x)

    def predict_direction(self, x):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
            confidence, direction = torch.max(probs, dim=1)
        return direction, confidence

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = BASE_DIR / 'data' / 'cache'
LONG_MODEL_DIR = BASE_DIR / 'models'
SHORT_MODEL_DIR = BASE_DIR / 'models_short'
META_MODEL_DIR = BASE_DIR / 'models_meta'

LONG_MODEL_DIR.mkdir(parents=True, exist_ok=True)
SHORT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
META_MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Shared params
# ============================================================
SEQUENCE_LENGTH = 30
SHORT_SEQUENCE_LENGTH = 30
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
LONG_FIXED_EPOCHS = 60     # Best epoch 57 from dev + buffer
LONG_BATCH_SIZE = 64
LONG_LR = 0.0015
LONG_NOISE_STD = 0.015
LONG_LABEL_SMOOTHING = 0.1
LONG_AUGMENT_COPIES = 2    # 3x total (original + 2 copies)
LONG_TEMPERATURE = 1.8807  # From FIL dev calibrated run

# ============================================================
# SHORT CNN params
# ============================================================
SHORT_FIXED_EPOCHS = 17    # Best epoch 14 from dev + buffer
SHORT_BATCH_SIZE = 32
SHORT_LR = 0.001
SHORT_NOISE_STD = 0.02
SHORT_LABEL_SMOOTHING = 0.05
SHORT_AUGMENT_COPIES = 3   # 4x total
SHORT_TEMPERATURE = 1.9515 # From FIL dev calibrated run


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
        logits = model(X_b)
        loss = criterion(logits, y_b)
        if torch.isnan(loss):
            logger.warning(f"  NaN loss detected! logits range: [{logits.min():.4f}, {logits.max():.4f}], "
                         f"any NaN in X: {torch.isnan(X_b).any()}, any NaN in logits: {torch.isnan(logits).any()}")
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()
        _, pred = torch.max(logits, 1)
        correct += (pred == y_b).sum().item()
        total += y_b.size(0)
    return total_loss / max(len(dataloader), 1), correct / max(total, 1)


# ============================================================
# STEP 1: Train LONG CNN on full data
# ============================================================
def train_long_cnn():
    logger.info(f"\n{'='*70}")
    logger.info(f"STEP 1: LONG CNN - Production Retrain (ALL data, {LONG_FIXED_EPOCHS} epochs)")
    logger.info(f"{'='*70}\n")

    df = pd.read_csv(DATA_DIR / 'fil_features.csv')
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
    }, LONG_MODEL_DIR / 'FIL_direction_model.pt')

    size_kb = (LONG_MODEL_DIR / 'FIL_direction_model.pt').stat().st_size / 1024
    logger.info(f"Saved: {LONG_MODEL_DIR / 'FIL_direction_model.pt'} ({size_kb:.0f} KB)")
    logger.info(f"Temperature (hardcoded): {LONG_TEMPERATURE}")

    return model, FEATURE_DIM


# ============================================================
# STEP 2: Train SHORT CNN on full data (DeepCNNShortModel)
# ============================================================
def train_short_cnn():
    logger.info(f"\n{'='*70}")
    logger.info(f"STEP 2: SHORT CNN - Production Retrain (ALL data, {SHORT_FIXED_EPOCHS} epochs)")
    logger.info(f"{'='*70}\n")

    df = pd.read_csv(DATA_DIR / 'fil_features.csv')
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
    X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = RobustScaler()
    X_scaled = np.clip(
        np.nan_to_num(scaler.fit_transform(X_raw), nan=0, posinf=0, neginf=0),
        -5, 5
    ).astype(np.float32)
    # Final safety check
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    logger.info(f"Data sanity: any NaN={np.isnan(X_scaled).any()}, any Inf={np.isinf(X_scaled).any()}")
    joblib.dump(scaler, SHORT_MODEL_DIR / 'feature_scaler_short.joblib')

    # Build sequences on ALL data
    y_raw = df['label'].values
    seqs, labels = [], []
    for i in range(SHORT_SEQUENCE_LENGTH, len(X_scaled)):
        if y_raw[i] != -1:
            seqs.append(X_scaled[i - SHORT_SEQUENCE_LENGTH:i])
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

    # Model - CNNDirectionModel for FIL SHORT (matches dev training)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNNDirectionModel(feature_dim=FEATURE_DIM, sequence_length=SHORT_SEQUENCE_LENGTH, dropout=0.4).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: CNNDirectionModel (SHORT, seq={SHORT_SEQUENCE_LENGTH}) | Params: {n_params:,} | Device: {device}")

    criterion = nn.CrossEntropyLoss(
        label_smoothing=SHORT_LABEL_SMOOTHING
    )
    optimizer = optim.AdamW(model.parameters(), lr=SHORT_LR, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)

    # Sanity check: test forward pass
    model.eval()
    with torch.no_grad():
        test_x = torch.FloatTensor(train_seqs_aug[:2]).to(device)
        test_out = model(test_x)
        logger.info(f"Sanity check: test output={test_out}, any NaN={torch.isnan(test_out).any()}")
        if torch.isnan(test_out).any():
            logger.error("Model produces NaN on forward pass! Reinitializing with different seed...")
            torch.manual_seed(42)
            model = CNNDirectionModel(feature_dim=FEATURE_DIM, sequence_length=SHORT_SEQUENCE_LENGTH, dropout=0.4).to(device)
            test_out2 = model(test_x)
            logger.info(f"After reinit: test output={test_out2}, any NaN={torch.isnan(test_out2).any()}")
            optimizer = optim.AdamW(model.parameters(), lr=SHORT_LR, weight_decay=5e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    model.train()

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
        'sequence_length': SHORT_SEQUENCE_LENGTH,
        'model_type': 'cnn_direction',
        'temperature': SHORT_TEMPERATURE,
        'short_atr_tp_mult': SHORT_ATR_TP_MULT,
        'short_atr_sl_mult': SHORT_ATR_SL_MULT,
        'production_retrain': True,
        'epochs_trained': SHORT_FIXED_EPOCHS,
    }, SHORT_MODEL_DIR / 'FIL_short_model.pt')

    # Save feature list
    with open(SHORT_MODEL_DIR / 'short_features.json', 'w') as f:
        json.dump(feature_cols, f)

    size_kb = (SHORT_MODEL_DIR / 'FIL_short_model.pt').stat().st_size / 1024
    logger.info(f"Saved: {SHORT_MODEL_DIR / 'FIL_short_model.pt'} ({size_kb:.0f} KB)")
    logger.info(f"Temperature (hardcoded): {SHORT_TEMPERATURE}")
    logger.info(f"Features saved: {SHORT_MODEL_DIR / 'short_features.json'}")

    return model, FEATURE_DIM, feature_cols


# ============================================================
# STEP 3: Retrain Meta XGBoost on full data
# ============================================================
def simulate_long_outcome(df, i, atr_val):
    """Check if LONG trade at index i would hit TP or SL"""
    if i >= len(df) - 2:
        return -1
    entry = df.iloc[i]['close']
    if pd.notna(atr_val) and atr_val > 0:
        tp = entry + atr_val * ATR_TP_MULT
        sl = entry - atr_val * ATR_SL_MULT
    else:
        tp = entry * (1 + FIXED_TP_PCT)
        sl = entry * (1 - FIXED_SL_PCT)
    for j in range(i + 1, min(i + 1 + BASE_LOOKAHEAD, len(df))):
        if df.iloc[j]['high'] >= tp:
            return 1
        if df.iloc[j]['low'] <= sl:
            return 0
    return -1


def simulate_short_outcome(df, i, atr_val):
    """Check if SHORT trade at index i would hit TP or SL"""
    if i >= len(df) - 2:
        return -1
    entry = df.iloc[i]['close']
    if pd.notna(atr_val) and atr_val > 0:
        tp = entry - atr_val * ATR_TP_MULT
        sl = entry + atr_val * ATR_SL_MULT
    else:
        tp = entry * (1 - FIXED_TP_PCT)
        sl = entry * (1 + FIXED_SL_PCT)
    for j in range(i + 1, min(i + 1 + BASE_LOOKAHEAD, len(df))):
        if df.iloc[j]['low'] <= tp:
            return 1
        if df.iloc[j]['high'] >= sl:
            return 0
    return -1


def build_meta_features(row, long_conf, long_dir, short_conf, short_dir,
                        long_prob_0, long_prob_1, short_prob_0, short_prob_1):
    """Build feature vector for meta-model from CNN outputs + market context"""
    feat = {
        'long_conf': long_conf,
        'long_dir': long_dir,
        'short_conf': short_conf,
        'short_dir': short_dir,
        'long_prob_spread': long_prob_1 - long_prob_0,
        'short_prob_spread': short_prob_1 - short_prob_0,
        'models_agree_bull': int(long_dir == 1 and short_dir == 0),
        'models_agree_bear': int(long_dir == 0 and short_dir == 1),
        'models_conflict': int(long_dir == 1 and short_dir == 1),
        'models_neutral': int(long_dir == 0 and short_dir == 0),
        'conf_diff': long_conf - short_conf,
        'max_conf': max(long_conf, short_conf),
        'min_conf': min(long_conf, short_conf),
    }

    market_cols = [
        '1d_rsi_14', '1d_adx_14', '1d_atr_14', '1d_macd_histogram',
        '1d_bb_width', '1d_stoch_k', '1d_cmf_20',
        'volatility_regime', 'volume_trend', 'trend_score',
        'distance_from_sma20', 'distance_from_sma50',
        'price_position_20', 'price_position_50',
        'regime_bull', 'regime_bear', 'regime_range',
        'accumulation_score', 'distribution_score',
        'vwap_trend_10', 'pressure_ratio',
        'trend_consistency_10', 'trend_consistency_20',
        'resistance_dist_pct', 'support_dist_pct',
        'sma50_above_sma200', 'sma_spread_pct',
        'rsi_bullish_count', 'macd_bullish_count',
        'adx_mean', 'momentum_bullish_count',
        'consecutive_up', 'consecutive_down',
        'body_ratio', 'day_of_week',
    ]

    for col in market_cols:
        val = row.get(col, np.nan) if col in row.index else np.nan
        feat[col] = val if pd.notna(val) else 0.0

    return feat


def train_meta(long_model, short_model):
    """Retrain meta XGBoost on full data using production CNN models."""
    from xgboost import XGBClassifier

    logger.info(f"\n{'='*70}")
    logger.info(f"STEP 3: META XGBOOST - Production Retrain (ALL data)")
    logger.info(f"{'='*70}\n")

    # Load features
    with open(BASE_DIR / 'required_features.json') as f:
        long_feature_cols = json.load(f)
    with open(SHORT_MODEL_DIR / 'short_features.json') as f:
        short_feature_cols = json.load(f)

    # Load data
    df = pd.read_csv(DATA_DIR / 'fil_features.csv')
    df['date'] = pd.to_datetime(df['date'])

    # Add bear features for SHORT
    df = add_bear_features(df)

    # Load scalers (just saved by production retrain)
    long_scaler = joblib.load(LONG_MODEL_DIR / 'feature_scaler.joblib')
    short_scaler = joblib.load(SHORT_MODEL_DIR / 'feature_scaler_short.joblib')

    # ATR for outcome simulation
    atr_series = ta_lib.volatility.AverageTrueRange(
        df['high'], df['low'], df['close'], window=14
    ).average_true_range()

    logger.info(f"Data: {len(df)} rows, {df['date'].min().date()} to {df['date'].max().date()}")

    # Scale features
    long_raw = df[long_feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    long_feat = np.clip(np.nan_to_num(long_scaler.transform(long_raw), nan=0, posinf=0, neginf=0), -5, 5)

    for c in short_feature_cols:
        if c not in df.columns:
            df[c] = 0
    short_raw = df[short_feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    short_feat = np.clip(np.nan_to_num(short_scaler.transform(short_raw), nan=0, posinf=0, neginf=0), -5, 5)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    long_model.eval()
    short_model.eval()

    seq = max(SEQUENCE_LENGTH, SHORT_SEQUENCE_LENGTH)

    # Generate predictions + outcomes for ALL days
    logger.info(f"Generating CNN predictions on full dataset...")
    meta_rows = []

    for i in range(seq, len(df)):
        row = df.iloc[i]
        atr_val = atr_series.iloc[i]

        # CNN LONG prediction
        lx = torch.tensor(long_feat[i - SEQUENCE_LENGTH:i], dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            l_logits = long_model(lx)
            l_probs = torch.softmax(l_logits / LONG_TEMPERATURE, dim=1).squeeze()
        l_conf, l_dir = l_probs.max(0)
        l_dir, l_conf = l_dir.item(), l_conf.item()
        l_p0, l_p1 = l_probs[0].item(), l_probs[1].item()

        # CNN SHORT prediction
        sx = torch.tensor(short_feat[i - SHORT_SEQUENCE_LENGTH:i], dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            s_logits = short_model(sx)
            s_probs = torch.softmax(s_logits / SHORT_TEMPERATURE, dim=1).squeeze()
        s_conf, s_dir = s_probs.max(0)
        s_dir, s_conf = s_dir.item(), s_conf.item()
        s_p0, s_p1 = s_probs[0].item(), s_probs[1].item()

        # Simulate outcomes
        long_outcome = simulate_long_outcome(df, i, atr_val)
        short_outcome = simulate_short_outcome(df, i, atr_val)

        mf = build_meta_features(row, l_conf, l_dir, s_conf, s_dir, l_p0, l_p1, s_p0, s_p1)
        mf['date'] = row['date']
        mf['long_outcome'] = long_outcome
        mf['short_outcome'] = short_outcome
        meta_rows.append(mf)

    meta_df = pd.DataFrame(meta_rows)
    logger.info(f"Generated {len(meta_df)} prediction rows")

    feature_cols_meta = [c for c in meta_df.columns if c not in ['date', 'long_outcome', 'short_outcome']]

    # === LONG META ===
    logger.info(f"\n--- LONG META-MODEL ---")
    long_mask = (meta_df['long_dir'] == 1) & (meta_df['long_outcome'] != -1)
    lt = meta_df[long_mask]
    logger.info(f"LONG meta: {len(lt)} samples (all data, no split)")

    if len(lt) > 50:
        X_lt = lt[feature_cols_meta].fillna(0).values
        y_lt = lt['long_outcome'].values
        n_win = (y_lt == 1).sum()
        n_lose = (y_lt == 0).sum()
        logger.info(f"  Dist: Win={n_win} ({n_win/len(y_lt)*100:.0f}%), Lose={n_lose} ({n_lose/len(y_lt)*100:.0f}%)")

        xgb_long = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_weight=5,
            gamma=1,
            reg_alpha=0.5,
            reg_lambda=2,
            scale_pos_weight=n_lose / max(n_win, 1),
            eval_metric='logloss',
            random_state=42,
            verbosity=0,
        )
        # Train on ALL data - no early stopping (fixed n_estimators)
        xgb_long.fit(X_lt, y_lt)
        joblib.dump(xgb_long, META_MODEL_DIR / 'FIL_meta_long.joblib')
        size_kb = (META_MODEL_DIR / 'FIL_meta_long.joblib').stat().st_size / 1024
        logger.info(f"  Saved: FIL_meta_long.joblib ({size_kb:.0f} KB)")
    else:
        logger.warning(f"  Not enough LONG samples ({len(lt)}), skipping meta LONG")

    # === SHORT META ===
    logger.info(f"\n--- SHORT META-MODEL ---")
    short_mask = (meta_df['short_dir'] == 1) & (meta_df['short_outcome'] != -1)
    st = meta_df[short_mask]
    logger.info(f"SHORT meta: {len(st)} samples (all data, no split)")

    if len(st) > 50:
        X_st = st[feature_cols_meta].fillna(0).values
        y_st = st['short_outcome'].values
        n_win = (y_st == 1).sum()
        n_lose = (y_st == 0).sum()
        logger.info(f"  Dist: Win={n_win} ({n_win/len(y_st)*100:.0f}%), Lose={n_lose} ({n_lose/len(y_st)*100:.0f}%)")

        xgb_short = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_weight=5,
            gamma=1,
            reg_alpha=0.5,
            reg_lambda=2,
            scale_pos_weight=n_lose / max(n_win, 1),
            eval_metric='logloss',
            random_state=42,
            verbosity=0,
        )
        xgb_short.fit(X_st, y_st)
        joblib.dump(xgb_short, META_MODEL_DIR / 'FIL_meta_short.joblib')
        size_kb = (META_MODEL_DIR / 'FIL_meta_short.joblib').stat().st_size / 1024
        logger.info(f"  Saved: FIL_meta_short.joblib ({size_kb:.0f} KB)")
    else:
        logger.warning(f"  Not enough SHORT samples ({len(st)}), skipping meta SHORT")

    # Save meta feature list
    with open(META_MODEL_DIR / 'meta_features.json', 'w') as f:
        json.dump(feature_cols_meta, f, indent=2)

    logger.info(f"  Meta features: {len(feature_cols_meta)}")


# ============================================================
# MAIN
# ============================================================
def main():
    logger.info(f"\n{'#'*70}")
    logger.info(f"#  FIL PRODUCTION RETRAIN - CNN + META MODELS ON FULL DATASET")
    logger.info(f"#  No train/val split | Fixed epochs | Hardcoded temperatures")
    logger.info(f"{'#'*70}\n")

    # Step 1: LONG CNN
    long_model, long_dim = train_long_cnn()

    # Step 2: SHORT CNN
    short_model, short_dim, short_features = train_short_cnn()

    # Step 3: Meta XGBoost (retrain on full data with new CNN models)
    train_meta(long_model, short_model)

    logger.info(f"\n{'#'*70}")
    logger.info(f"#  PRODUCTION RETRAIN COMPLETE")
    logger.info(f"{'#'*70}")
    logger.info(f"  LONG CNN:       {LONG_MODEL_DIR / 'FIL_direction_model.pt'}")
    logger.info(f"  LONG scaler:    {LONG_MODEL_DIR / 'feature_scaler.joblib'}")
    logger.info(f"  SHORT CNN:      {SHORT_MODEL_DIR / 'FIL_short_model.pt'}")
    logger.info(f"  SHORT scaler:   {SHORT_MODEL_DIR / 'feature_scaler_short.joblib'}")
    logger.info(f"  SHORT features: {SHORT_MODEL_DIR / 'short_features.json'}")
    logger.info(f"  Meta LONG:      {META_MODEL_DIR / 'FIL_meta_long.joblib'}")
    logger.info(f"  Meta SHORT:     {META_MODEL_DIR / 'FIL_meta_short.joblib'}")
    logger.info(f"  Meta features:  {META_MODEL_DIR / 'meta_features.json'}")
    logger.info(f"\n  Temperatures: LONG={LONG_TEMPERATURE}, SHORT={SHORT_TEMPERATURE}")
    logger.info(f"  Epochs: LONG={LONG_FIXED_EPOCHS}, SHORT={SHORT_FIXED_EPOCHS}")


if __name__ == "__main__":
    main()
