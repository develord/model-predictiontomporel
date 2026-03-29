"""
BTC PyTorch Training Script - V2 Lightweight
=============================================
Uses LightweightDirectionModel (GRU) instead of heavy Transformer+LSTM.
Much better suited for ~2500 training samples.

Fixes:
- Lightweight GRU model (~200K params vs 13M)
- StandardScaler normalization
- Class weights for imbalanced labels
- Gradient clipping
- Higher learning rate for smaller model
- Sequence length 30 (more training samples)

Usage:
    python 03_train_pytorch_model.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import logging
import json
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR / 'scripts'))

from direction_prediction_model import LightweightDirectionModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = BASE_DIR / 'data' / 'cache'
MODEL_DIR = BASE_DIR / 'models'
RESULTS_DIR = BASE_DIR / 'results'
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Training parameters
SEQUENCE_LENGTH = 30
FEATURE_DIM = 90
BATCH_SIZE = 64
EPOCHS = 200
LEARNING_RATE = 0.001
PATIENCE = 30
GRAD_CLIP = 1.0

# Triple barrier labeling parameters
TP_PCT = 0.015  # 1.5% take profit
SL_PCT = 0.0075  # 0.75% stop loss

# Train/Val split (exclude Q1 2026 for test)
TRAIN_START = '2018-01-01'
TRAIN_END = '2024-12-31'
VAL_START = '2025-01-01'
VAL_END = '2025-12-31'


class CryptoSequenceDataset(Dataset):
    def __init__(self, features, labels, sequence_length=30):
        self.features = features
        self.labels = labels
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.features) - self.sequence_length

    def __getitem__(self, idx):
        seq = self.features[idx:idx+self.sequence_length]
        label = self.labels[idx+self.sequence_length]
        return torch.FloatTensor(seq), torch.LongTensor([label])[0]


def create_labels(df):
    """Create labels using triple barrier method"""
    labels = []
    for i in range(len(df)):
        if i == len(df) - 1:
            labels.append(-1)
            continue

        entry_price = df.iloc[i]['close']
        tp_price = entry_price * (1 + TP_PCT)
        sl_price = entry_price * (1 - SL_PCT)

        hit_tp = False
        hit_sl = False

        for j in range(i+1, min(i+11, len(df))):
            high = df.iloc[j]['high']
            low = df.iloc[j]['low']

            if high >= tp_price:
                hit_tp = True
                break
            if low <= sl_price:
                hit_sl = True
                break

        if hit_tp:
            labels.append(1)
        elif hit_sl:
            labels.append(0)
        else:
            labels.append(-1)

    return np.array(labels)


def train_epoch(model, dataloader, criterion, optimizer, device, grad_clip):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for features, labels in tqdm(dataloader, desc="Training"):
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(dataloader), correct / total


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device)

            logits = model(features)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / len(dataloader), correct / total, all_preds, all_labels


def train_model():
    logger.info(f"\n{'='*70}")
    logger.info(f"BTC LIGHTWEIGHT GRU MODEL TRAINING")
    logger.info(f"{'='*70}\n")

    # Load features
    features_file = DATA_DIR / 'btc_features.csv'
    if not features_file.exists():
        logger.error(f"Features file not found: {features_file}")
        return

    df = pd.read_csv(features_file)
    df['date'] = pd.to_datetime(df['date'])

    logger.info(f"Loaded {len(df)} candles")
    logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")

    with open(BASE_DIR / 'required_features.json', 'r') as f:
        feature_cols = json.load(f)

    logger.info(f"Using {len(feature_cols)} features")

    # Create labels
    logger.info(f"\nCreating labels (TP={TP_PCT:.1%}, SL={SL_PCT:.1%})...")
    df['label'] = create_labels(df)

    df_labeled = df[df['label'] != -1].copy()
    logger.info(f"Labeled samples: {len(df_labeled)}")

    label_counts = df_labeled['label'].value_counts()
    n_short = label_counts.get(0, 0)
    n_long = label_counts.get(1, 0)
    logger.info(f"Class distribution: SHORT={n_short} ({n_short/len(df_labeled)*100:.1f}%), LONG={n_long} ({n_long/len(df_labeled)*100:.1f}%)")

    # Class weights
    total_samples = n_short + n_long
    weight_short = total_samples / (2 * n_short) if n_short > 0 else 1.0
    weight_long = total_samples / (2 * n_long) if n_long > 0 else 1.0
    class_weights = torch.FloatTensor([weight_short, weight_long])
    logger.info(f"Class weights: SHORT={weight_short:.3f}, LONG={weight_long:.3f}")

    # Split train/val using FULL dataframe (not just labeled) for proper sequences
    train_mask = (df['date'] >= TRAIN_START) & (df['date'] <= TRAIN_END)
    val_mask = (df['date'] >= VAL_START) & (df['date'] <= VAL_END)

    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()

    logger.info(f"\nTrain period: {len(train_df)} rows ({train_df['date'].min().date()} to {train_df['date'].max().date()})")
    logger.info(f"Val period: {len(val_df)} rows ({val_df['date'].min().date()} to {val_df['date'].max().date()})")

    # Prepare features with StandardScaler
    X_train_raw = train_df[feature_cols].fillna(0).values
    X_val_raw = val_df[feature_cols].fillna(0).values
    y_train = train_df['label'].values
    y_val = val_df['label'].values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)

    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)

    # Save scaler
    scaler_path = MODEL_DIR / 'feature_scaler.joblib'
    joblib.dump(scaler, scaler_path)
    logger.info(f"Scaler saved to {scaler_path}")
    logger.info(f"Feature stats: mean={X_train.mean():.4f}, std={X_train.std():.4f}")

    # Filter sequences - only keep those where the target label is valid (not -1)
    # Build datasets with sequences that have valid labels
    train_sequences = []
    train_labels = []
    for i in range(SEQUENCE_LENGTH, len(X_train)):
        if y_train[i] != -1:
            train_sequences.append(X_train[i-SEQUENCE_LENGTH:i])
            train_labels.append(y_train[i])

    val_sequences = []
    val_labels = []
    for i in range(SEQUENCE_LENGTH, len(X_val)):
        if y_val[i] != -1:
            val_sequences.append(X_val[i-SEQUENCE_LENGTH:i])
            val_labels.append(y_val[i])

    train_sequences = np.array(train_sequences)
    train_labels = np.array(train_labels)
    val_sequences = np.array(val_sequences)
    val_labels = np.array(val_labels)

    logger.info(f"\nValid train sequences: {len(train_sequences)}")
    logger.info(f"Valid val sequences: {len(val_sequences)}")
    logger.info(f"Train label dist: SHORT={sum(train_labels==0)}, LONG={sum(train_labels==1)}")
    logger.info(f"Val label dist: SHORT={sum(val_labels==0)}, LONG={sum(val_labels==1)}")

    # Create tensor datasets directly
    train_X_tensor = torch.FloatTensor(train_sequences)
    train_y_tensor = torch.LongTensor(train_labels)
    val_X_tensor = torch.FloatTensor(val_sequences)
    val_y_tensor = torch.LongTensor(val_labels)

    train_dataset = torch.utils.data.TensorDataset(train_X_tensor, train_y_tensor)
    val_dataset = torch.utils.data.TensorDataset(val_X_tensor, val_y_tensor)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Create lightweight model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"\nDevice: {device}")

    model = LightweightDirectionModel(
        feature_dim=FEATURE_DIM,
        sequence_length=SEQUENCE_LENGTH,
        hidden_dim=128,
        num_layers=2,
        dropout=0.3
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: LightweightDirectionModel (GRU)")
    logger.info(f"Parameters: {n_params:,}")

    # Loss with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

    logger.info(f"\nTraining: {EPOCHS} epochs, LR={LEARNING_RATE}, batch={BATCH_SIZE}")
    logger.info(f"Gradient clipping: {GRAD_CLIP}, Patience: {PATIENCE}\n")

    best_val_loss = float('inf')
    best_val_acc = 0
    patience_counter = 0
    best_epoch = 0

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, GRAD_CLIP)
        val_loss, val_acc, val_preds, val_labels_list = validate(model, val_loader, criterion, device)

        scheduler.step(epoch)

        preds_array = np.array(val_preds)
        n_buy = (preds_array == 1).sum()
        n_sell = (preds_array == 0).sum()

        logger.info(f"Epoch {epoch+1}/{EPOCHS} | "
                     f"Train: loss={train_loss:.4f} acc={train_acc:.4f} | "
                     f"Val: loss={val_loss:.4f} acc={val_acc:.4f} | "
                     f"Preds: BUY={n_buy} SELL={n_sell} | "
                     f"LR={optimizer.param_groups[0]['lr']:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch + 1
            patience_counter = 0

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'feature_dim': FEATURE_DIM,
                'sequence_length': SEQUENCE_LENGTH,
                'model_type': 'lightweight'
            }, MODEL_DIR / 'BTC_direction_model.pt')

            logger.info(f"  >>> Best model saved (loss={val_loss:.4f}, acc={val_acc:.4f})")
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            logger.info(f"\nEarly stopping at epoch {epoch+1}")
            break

    # Load best model
    checkpoint = torch.load(MODEL_DIR / 'BTC_direction_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    logger.info(f"\n{'='*70}")
    logger.info(f"TRAINING COMPLETE")
    logger.info(f"{'='*70}\n")
    logger.info(f"Best epoch: {best_epoch}")
    logger.info(f"Best val loss: {best_val_loss:.4f}")
    logger.info(f"Best val acc: {best_val_acc:.4f}")

    _, _, val_preds, val_labels_final = validate(model, val_loader, criterion, device)

    logger.info(f"\n{classification_report(val_labels_final, val_preds, target_names=['SHORT', 'LONG'])}")

    cm = confusion_matrix(val_labels_final, val_preds)
    logger.info(f"Confusion Matrix:")
    logger.info(f"  Predicted SHORT | Predicted LONG")
    logger.info(f"Actual SHORT: {cm[0][0]:4d} | {cm[0][1]:4d}")
    logger.info(f"Actual LONG:  {cm[1][0]:4d} | {cm[1][1]:4d}")

    size_mb = (MODEL_DIR / 'BTC_direction_model.pt').stat().st_size / (1024*1024)
    logger.info(f"\nModel saved: {size_mb:.2f} MB")


if __name__ == "__main__":
    train_model()
