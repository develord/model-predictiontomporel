"""
BTC SHORT CNN Training Script
===============================
Train CNN to detect SHORT opportunities (price drops).

Labels:
  1 = price drops >= 1.5% within 10 days (SHORT profitable)
  0 = price doesn't drop enough (no SHORT)
 -1 = unclear / no signal

Same architecture as LONG model but inverted labels.

Usage:
    python 03_train_short_model.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import logging
import json
import joblib
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR / 'scripts'))
from direction_prediction_model import CNNDirectionModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = BASE_DIR / 'data' / 'cache'
MODEL_DIR = BASE_DIR / 'models_short'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SEQUENCE_LENGTH = 30
BATCH_SIZE = 64
EPOCHS = 200
LEARNING_RATE = 0.0015
PATIENCE = 35
GRAD_CLIP = 0.5
NOISE_STD = 0.015
LABEL_SMOOTHING = 0.1

# SHORT labels: price drops by SL_PCT within 10 days
SHORT_TP_PCT = 0.015   # 1.5% drop = profitable short
SHORT_SL_PCT = 0.0075  # 0.75% rise = stop loss on short

TRAIN_END = '2024-12-31'
VAL_START = '2025-01-01'
VAL_END = '2025-12-31'


def create_short_labels(df):
    """Create SHORT labels: 1 = price drops enough, 0 = doesn't drop"""
    labels = []
    for i in range(len(df)):
        if i >= len(df) - 1:
            labels.append(-1)
            continue

        entry = df.iloc[i]['close']
        # For SHORT: TP when price drops, SL when price rises
        tp_price = entry * (1 - SHORT_TP_PCT)   # Price target (lower)
        sl_price = entry * (1 + SHORT_SL_PCT)    # Stop loss (higher)

        hit_tp, hit_sl = False, False

        for j in range(i+1, min(i+11, len(df))):
            if df.iloc[j]['low'] <= tp_price:
                hit_tp = True
                break
            if df.iloc[j]['high'] >= sl_price:
                hit_sl = True
                break

        if hit_tp:
            labels.append(1)   # SHORT profitable
        elif hit_sl:
            labels.append(0)   # SHORT would lose
        else:
            labels.append(-1)  # Unclear

    return np.array(labels)


def build_sequences(X, y, seq_len):
    sequences, targets = [], []
    for i in range(seq_len, len(X)):
        if y[i] != -1:
            sequences.append(X[i-seq_len:i])
            targets.append(y[i])
    return np.array(sequences), np.array(targets)


def augment_data(X, y, noise_std=0.015, n_copies=2):
    aug_X, aug_y = [X], [y]
    for _ in range(n_copies):
        aug_X.append(X + np.random.normal(0, noise_std, X.shape).astype(np.float32))
        aug_y.append(y)
    return np.concatenate(aug_X), np.concatenate(aug_y)


def train():
    logger.info(f"\n{'='*70}")
    logger.info(f"BTC SHORT CNN TRAINING")
    logger.info(f"{'='*70}\n")

    # Load features (same as LONG model)
    df = pd.read_csv(DATA_DIR / 'btc_features.csv')
    df['date'] = pd.to_datetime(df['date'])
    logger.info(f"Data: {len(df)} rows")

    with open(BASE_DIR / 'required_features.json') as f:
        feature_cols = json.load(f)
    FEATURE_DIM = len(feature_cols)
    logger.info(f"Features: {FEATURE_DIM}")

    # Create SHORT labels
    logger.info("Creating SHORT labels...")
    df['label'] = create_short_labels(df)

    n_short_ok = (df['label'] == 1).sum()
    n_short_no = (df['label'] == 0).sum()
    n_unclear = (df['label'] == -1).sum()
    logger.info(f"  SHORT profitable (1): {n_short_ok}")
    logger.info(f"  SHORT would lose (0): {n_short_no}")
    logger.info(f"  Unclear (-1): {n_unclear}")

    # Split
    train_df = df[df['date'] <= TRAIN_END].copy()
    val_df = df[(df['date'] >= VAL_START) & (df['date'] <= VAL_END)].copy()
    logger.info(f"Train: {len(train_df)} | Val: {len(val_df)}")

    # Scale
    X_train_raw = train_df[feature_cols].fillna(0).values.astype(np.float32)
    X_val_raw = val_df[feature_cols].fillna(0).values.astype(np.float32)
    y_train_raw = train_df['label'].values
    y_val_raw = val_df['label'].values

    scaler = RobustScaler()
    X_train = np.clip(np.nan_to_num(scaler.fit_transform(X_train_raw), nan=0, posinf=0, neginf=0), -5, 5).astype(np.float32)
    X_val = np.clip(np.nan_to_num(scaler.transform(X_val_raw), nan=0, posinf=0, neginf=0), -5, 5).astype(np.float32)

    joblib.dump(scaler, MODEL_DIR / 'feature_scaler_short.joblib')

    # Build sequences
    train_seqs, train_labels = build_sequences(X_train, y_train_raw, SEQUENCE_LENGTH)
    val_seqs, val_labels = build_sequences(X_val, y_val_raw, SEQUENCE_LENGTH)

    n_0 = (train_labels == 0).sum()
    n_1 = (train_labels == 1).sum()
    logger.info(f"Train seq: {len(train_seqs)} (NoShort={n_0}, Short={n_1}) | Val: {len(val_seqs)}")

    # Augment
    train_seqs_aug, train_labels_aug = augment_data(train_seqs, train_labels, NOISE_STD, n_copies=2)
    logger.info(f"After augmentation: {len(train_seqs_aug)}")

    # Class weights
    w_0 = len(train_labels) / (2 * n_0) if n_0 > 0 else 1.0
    w_1 = len(train_labels) / (2 * n_1) if n_1 > 0 else 1.0
    class_weights = torch.FloatTensor([w_0, w_1])

    train_loader = DataLoader(TensorDataset(torch.FloatTensor(train_seqs_aug), torch.LongTensor(train_labels_aug)),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.FloatTensor(val_seqs), torch.LongTensor(val_labels)),
                            batch_size=BATCH_SIZE)

    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNNDirectionModel(feature_dim=FEATURE_DIM, sequence_length=SEQUENCE_LENGTH, dropout=0.4).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: CNNDirectionModel | Params: {n_params:,}")

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)

    best_val_acc, patience_counter, best_epoch = 0, 0, 0

    for epoch in range(EPOCHS):
        # Train
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
        scheduler.step(epoch)

        # Validate
        model.eval()
        correct, total, n_short, n_no = 0, 0, 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).argmax(1)
                correct += (pred == yb).sum().item()
                total += yb.size(0)
                n_short += (pred == 1).sum().item()
                n_no += (pred == 0).sum().item()
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(yb.cpu().numpy())

        acc = correct / total if total > 0 else 0
        preds_arr = np.array(all_preds)
        labels_arr = np.array(all_labels)
        short_prec = labels_arr[preds_arr == 1].mean() * 100 if n_short > 0 else 0

        if (epoch + 1) % 5 == 0:
            logger.info(f"E{epoch+1:3d}/{EPOCHS} | Acc:{acc:.3f} | SHORT={n_short} NO={n_no} | ShortPrec={short_prec:.0f}%")

        is_better = acc > best_val_acc and n_short >= 5 and n_no >= 5
        if is_better:
            best_val_acc = acc
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'feature_dim': FEATURE_DIM,
                'sequence_length': SEQUENCE_LENGTH,
                'model_type': 'cnn',
                'direction': 'short',
                'epoch': epoch + 1,
                'val_acc': acc,
            }, MODEL_DIR / 'BTC_short_model.pt')
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    # Final evaluation
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

    all_dirs = np.array(all_dirs)
    all_confs = np.array(all_confs)
    all_trues = np.array(all_trues)

    logger.info(f"\n{'='*70}")
    logger.info(f"TRAINING COMPLETE | Best epoch: {best_epoch} | Best acc: {best_val_acc:.3f}")
    logger.info(f"{'='*70}")

    logger.info(f"\nConfidence Analysis (SHORT signals):")
    for thresh in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
        mask = (all_dirs == 1) & (all_confs >= thresh)
        if mask.sum() > 0:
            wr = all_trues[mask].mean() * 100
            logger.info(f"  Conf >= {thresh:.0%}: {mask.sum()} signals, WR: {wr:.1f}%")

    # Save feature list for SHORT model
    with open(MODEL_DIR / 'short_features.json', 'w') as f:
        json.dump(feature_cols, f)

    logger.info(f"\nModel: {(MODEL_DIR / 'BTC_short_model.pt').stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    train()
