"""
${coin} 1D-CNN Training Script
============================
Uses CNNDirectionModel with existing multi-TF features (384 features).
Same approach as BTC CNN which achieved 75% WR.

Usage:
    python 03_train_cnn.py
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
from sklearn.metrics import classification_report, confusion_matrix

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR / 'scripts'))

from direction_prediction_model import CNNDirectionModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = BASE_DIR / 'data' / 'cache'
MODEL_DIR = BASE_DIR / 'models'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

CRYPTO = 'XRP'
SEQUENCE_LENGTH = 30
BATCH_SIZE = 64
EPOCHS = 200
LEARNING_RATE = 0.0015
PATIENCE = 35
GRAD_CLIP = 0.5
NOISE_STD = 0.015
LABEL_SMOOTHING = 0.1

TP_PCT = 0.015
SL_PCT = 0.0075

TRAIN_END = '2024-12-31'
VAL_START = '2025-01-01'
VAL_END = '2025-12-31'


def create_labels(df):
    """Create triple barrier labels"""
    labels = []
    for i in range(len(df)):
        if i >= len(df) - 1:
            labels.append(-1)
            continue
        entry = df.iloc[i]['close']
        tp = entry * (1 + TP_PCT)
        sl = entry * (1 - SL_PCT)
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


def build_sequences(X, y, seq_len):
    sequences, targets = [], []
    for i in range(seq_len, len(X)):
        if y[i] != -1:
            sequences.append(X[i-seq_len:i])
            targets.append(y[i])
    return np.array(sequences), np.array(targets)


def augment_data(X, y, noise_std=0.015, n_copies=2):
    augmented_X = [X]
    augmented_y = [y]
    for _ in range(n_copies):
        noise = np.random.normal(0, noise_std, X.shape).astype(np.float32)
        augmented_X.append(X + noise)
        augmented_y.append(y)
    return np.concatenate(augmented_X), np.concatenate(augmented_y)


def train_epoch(model, dataloader, criterion, optimizer, device, grad_clip):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()
        _, pred = torch.max(logits, 1)
        correct += (pred == y_batch).sum().item()
        total += y_batch.size(0)
    return total_loss / len(dataloader), correct / total


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item()
            _, pred = torch.max(logits, 1)
            correct += (pred == y_batch).sum().item()
            total += y_batch.size(0)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    return total_loss / len(dataloader), correct / total, all_preds, all_labels


def train():
    logger.info(f"\n{'='*70}")
    logger.info(f"{CRYPTO} 1D-CNN TRAINING")
    logger.info(f"{'='*70}\n")

    # Load data
    df = pd.read_csv(DATA_DIR / f'xrp_features.csv')
    df['date'] = pd.to_datetime(df['date'])
    logger.info(f"Data: {len(df)} rows")

    # Feature columns - exclude leaks and meta
    exclude = ['date', 'timestamp', 'label_class', 'triple_barrier_label', 'label_numeric',
               'open', 'high', 'low', 'close', 'volume',
               'price_target_pct', 'future_price', 'future_return']
    feature_cols = [c for c in df.columns if c not in exclude
                    and 'future' not in c.lower() and 'target' not in c.lower()
                    and not c.startswith('Unnamed')]

    FEATURE_DIM = len(feature_cols)
    logger.info(f"Features: {FEATURE_DIM}")

    # Save feature list
    with open(MODEL_DIR / f'{CRYPTO.lower()}_cnn_features.json', 'w') as f:
        json.dump(feature_cols, f)

    # Labels - always recalculate with proper triple barrier
    if False:  # Always use our own labels
        df['label'] = df['triple_barrier_label'].replace(-1, 0)
        # Mark unlabeled rows
        df.loc[df['label_class'] == 'HOLD', 'label'] = -1
    else:
        df['label'] = create_labels(df)

    # Split
    train_df = df[df['date'] <= TRAIN_END].copy()
    val_df = df[(df['date'] >= VAL_START) & (df['date'] <= VAL_END)].copy()
    logger.info(f"Train: {len(train_df)} | Val: {len(val_df)}")

    # Scale
    X_train_raw = train_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    X_val_raw = val_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    y_train_raw = train_df['label'].values
    y_val_raw = val_df['label'].values

    scaler = RobustScaler()
    X_train = np.clip(np.nan_to_num(scaler.fit_transform(X_train_raw), nan=0, posinf=0, neginf=0), -5, 5).astype(np.float32)
    X_val = np.clip(np.nan_to_num(scaler.transform(X_val_raw), nan=0, posinf=0, neginf=0), -5, 5).astype(np.float32)

    joblib.dump(scaler, MODEL_DIR / 'feature_scaler.joblib')

    # Build sequences
    train_seqs, train_labels = build_sequences(X_train, y_train_raw, SEQUENCE_LENGTH)
    val_seqs, val_labels = build_sequences(X_val, y_val_raw, SEQUENCE_LENGTH)

    n_sl = (train_labels == 0).sum()
    n_tp = (train_labels == 1).sum()
    logger.info(f"Train seq: {len(train_seqs)} (SL={n_sl}, TP={n_tp}) | Val seq: {len(val_seqs)}")

    # Augment
    train_seqs_aug, train_labels_aug = augment_data(train_seqs, train_labels, NOISE_STD, n_copies=2)
    logger.info(f"After augmentation: {len(train_seqs_aug)}")

    # Class weights
    w_sl = len(train_labels) / (2 * n_sl) if n_sl > 0 else 1.0
    w_tp = len(train_labels) / (2 * n_tp) if n_tp > 0 else 1.0
    class_weights = torch.FloatTensor([w_sl, w_tp])

    # Dataloaders
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(train_seqs_aug), torch.LongTensor(train_labels_aug)),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.FloatTensor(val_seqs), torch.LongTensor(val_labels)),
                            batch_size=BATCH_SIZE, shuffle=False)

    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNNDirectionModel(feature_dim=FEATURE_DIM, sequence_length=SEQUENCE_LENGTH, dropout=0.4).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: CNNDirectionModel | Params: {n_params:,} | Device: {device}")

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)

    best_val_acc = 0
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, GRAD_CLIP)
        val_loss, val_acc, val_preds, val_labels_list = validate(model, val_loader, criterion, device)
        scheduler.step(epoch)

        preds = np.array(val_preds)
        labels_arr = np.array(val_labels_list)
        n_buy = (preds == 1).sum()
        n_sell = (preds == 0).sum()
        buy_prec = labels_arr[preds == 1].mean() * 100 if n_buy > 0 else 0

        logger.info(f"E{epoch+1:3d}/{EPOCHS} | T:{train_loss:.4f}/{train_acc:.3f} | "
                     f"V:{val_loss:.4f}/{val_acc:.3f} | BUY={n_buy} SELL={n_sell} | "
                     f"BuyP={buy_prec:.0f}% | LR={optimizer.param_groups[0]['lr']:.5f}")

        is_better = val_acc > best_val_acc and n_buy >= 10 and n_sell >= 10
        if is_better:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'feature_dim': FEATURE_DIM,
                'sequence_length': SEQUENCE_LENGTH,
                'model_type': 'cnn'
            }, MODEL_DIR / f'{CRYPTO.lower()}_cnn_model.pt')
            logger.info(f"  >>> BEST (acc={val_acc:.3f}, buyP={buy_prec:.0f}%)")
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            logger.info(f"\nEarly stopping at epoch {epoch+1}")
            break

    # Load best
    ckpt = torch.load(MODEL_DIR / f'{CRYPTO.lower()}_cnn_model.pt')
    model.load_state_dict(ckpt['model_state_dict'])

    _, _, final_preds, final_labels = validate(model, val_loader, criterion, device)

    logger.info(f"\n{'='*70}")
    logger.info(f"TRAINING COMPLETE | Best epoch: {best_epoch} | Best acc: {best_val_acc:.3f}")
    logger.info(f"{'='*70}")
    logger.info(f"\n{classification_report(final_labels, final_preds, target_names=['SL', 'TP'])}")

    # Confidence analysis
    model.eval()
    all_confs, all_dirs, all_trues = [], [], []
    with torch.no_grad():
        for X_b, y_b in val_loader:
            d, c = model.predict_direction(X_b.to(device))
            all_dirs.extend(d.cpu().numpy())
            all_confs.extend(c.cpu().numpy())
            all_trues.extend(y_b.numpy())

    all_dirs = np.array(all_dirs)
    all_confs = np.array(all_confs)
    all_trues = np.array(all_trues)

    logger.info(f"\nConfidence Analysis (BUY signals):")
    for thresh in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
        mask = (all_dirs == 1) & (all_confs >= thresh)
        if mask.sum() > 0:
            wr = all_trues[mask].mean() * 100
            logger.info(f"  Conf >= {thresh:.0%}: {mask.sum()} signals, WR: {wr:.1f}%")

    logger.info(f"\nModel: {(MODEL_DIR / f'{CRYPTO.lower()}_cnn_model.pt').stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    train()
