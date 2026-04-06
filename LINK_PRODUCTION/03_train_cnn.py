"""
LINK 1D-CNN Training Script V3
================================
Uses CNNDirectionModel - multi-scale convolutions + temporal attention.
Much fewer params than GRU, resistant to overfitting on small datasets.

Key design:
- Multi-scale convolutions (3,5,7) detect patterns at different horizons
- Temporal attention focuses on most predictive timesteps
- Heavy regularization: dropout + batch norm + weight decay
- Short sequences (30 days) = more training samples
- Data augmentation via noise injection
- Temperature calibration

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
from tqdm import tqdm

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR / 'scripts'))

from direction_prediction_model import CNNDirectionModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = BASE_DIR / 'data' / 'cache'
MODEL_DIR = BASE_DIR / 'models'
RESULTS_DIR = BASE_DIR / 'results'
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Params
SEQUENCE_LENGTH = 30  # Longer to capture macro trends
BATCH_SIZE = 64
EPOCHS = 200
LEARNING_RATE = 0.0015
PATIENCE = 35
GRAD_CLIP = 0.5
NOISE_STD = 0.015
LABEL_SMOOTHING = 0.1

# Match feature engineering ATR-based labeling
ATR_TP_MULT = 1.5
ATR_SL_MULT = 1.5
FIXED_TP_PCT = 0.012
FIXED_SL_PCT = 0.012
BASE_LOOKAHEAD = 10
ATR_LOOKAHEAD_MULT = 0.7

TRAIN_START = '2017-12-01'
TRAIN_END = '2025-06-30'
VAL_START = '2025-07-01'
VAL_END = '2025-12-31'


def create_labels(df):
    """ATR-adaptive labels — labels are kept but bear_flag used for sample weighting."""
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
        for j in range(i+1, min(i+1+lookahead, len(df))):
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
    """Build sequences, only keep valid labels"""
    sequences, targets = [], []
    for i in range(seq_len, len(X)):
        if y[i] != -1:
            sequences.append(X[i-seq_len:i])
            targets.append(y[i])
    return np.array(sequences), np.array(targets)


def augment_data(X, y, noise_std=0.02, n_copies=2):
    """Data augmentation: add noise copies of training data"""
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
    for batch in dataloader:
        if len(batch) == 3:
            X_batch, y_batch, w_batch = batch
            X_batch, y_batch, w_batch = X_batch.to(device), y_batch.to(device), w_batch.to(device)
        else:
            X_batch, y_batch = batch[0].to(device), batch[1].to(device)
            w_batch = None
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        # Apply sample weights
        if w_batch is not None:
            loss = (loss * w_batch).mean() if loss.dim() > 0 else loss
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
        for batch in dataloader:
            X_batch, y_batch = batch[0].to(device), batch[1].to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch).mean()
            total_loss += loss.item()
            _, pred = torch.max(logits, 1)
            correct += (pred == y_batch).sum().item()
            total += y_batch.size(0)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    return total_loss / len(dataloader), correct / total, all_preds, all_labels


def train():
    logger.info(f"\n{'='*70}")
    logger.info(f"LINK 1D-CNN TRAINING V3 (Multi-scale Conv + Attention)")
    logger.info(f"{'='*70}\n")

    df = pd.read_csv(DATA_DIR / 'link_features.csv')
    df['date'] = pd.to_datetime(df['date'])
    logger.info(f"Data: {len(df)} rows, {df['date'].min().date()} to {df['date'].max().date()}")

    with open(BASE_DIR / 'required_features.json') as f:
        feature_cols = json.load(f)
    FEATURE_DIM = len(feature_cols)
    logger.info(f"Features: {FEATURE_DIM}")

    # Labels
    if 'label' not in df.columns:
        df['label'] = create_labels(df)

    # Split by date
    train_mask = (df['date'] >= TRAIN_START) & (df['date'] <= TRAIN_END)
    val_mask = (df['date'] >= VAL_START) & (df['date'] <= VAL_END)
    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()

    logger.info(f"Train: {len(train_df)} | Val: {len(val_df)}")

    # Scale features with RobustScaler
    X_train_raw = train_df[feature_cols].fillna(0).values.astype(np.float32)
    X_val_raw = val_df[feature_cols].fillna(0).values.astype(np.float32)
    y_train_raw = train_df['label'].values
    y_val_raw = val_df['label'].values

    scaler = RobustScaler()
    X_train_scaled = np.nan_to_num(scaler.fit_transform(X_train_raw), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    X_val_scaled = np.nan_to_num(scaler.transform(X_val_raw), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # Clip extremes
    X_train_scaled = np.clip(X_train_scaled, -5, 5)
    X_val_scaled = np.clip(X_val_scaled, -5, 5)

    joblib.dump(scaler, MODEL_DIR / 'feature_scaler.joblib')

    # Build sequences with sample weights
    bear_flag = train_df['is_strong_bear'].values if 'is_strong_bear' in train_df.columns else np.zeros(len(train_df))

    train_seqs, train_labels, train_weights = [], [], []
    for i in range(SEQUENCE_LENGTH, len(X_train_scaled)):
        if y_train_raw[i] != -1:
            train_seqs.append(X_train_scaled[i-SEQUENCE_LENGTH:i])
            train_labels.append(y_train_raw[i])
            is_bear = bear_flag[i] if i < len(bear_flag) else 0
            if is_bear and y_train_raw[i] == 1:
                train_weights.append(0.3)
            elif is_bear and y_train_raw[i] == 0:
                train_weights.append(2.0)
            else:
                train_weights.append(1.0)
    train_seqs = np.array(train_seqs)
    train_labels = np.array(train_labels)
    train_weights = np.array(train_weights, dtype=np.float32)

    val_seqs, val_labels = build_sequences(X_val_scaled, y_val_raw, SEQUENCE_LENGTH)

    logger.info(f"Train sequences: {len(train_seqs)} | Val: {len(val_seqs)}")

    n_sl = (train_labels == 0).sum()
    n_tp = (train_labels == 1).sum()
    n_bear_tp = ((train_labels == 1) & (train_weights < 1.0)).sum()
    n_bear_sl = ((train_labels == 0) & (train_weights > 1.0)).sum()
    logger.info(f"Train dist: SL={n_sl} ({n_sl/len(train_labels)*100:.1f}%), TP={n_tp} ({n_tp/len(train_labels)*100:.1f}%)")
    logger.info(f"Bear-weighted: {n_bear_tp} TP downweighted (0.3x), {n_bear_sl} SL upweighted (2x)")

    # Data augmentation (keep weights aligned)
    aug_seqs, aug_labels, aug_weights = [train_seqs], [train_labels], [train_weights]
    for _ in range(2):
        noise = np.random.normal(0, NOISE_STD, train_seqs.shape).astype(np.float32)
        aug_seqs.append(train_seqs + noise)
        aug_labels.append(train_labels)
        aug_weights.append(train_weights)
    train_seqs_aug = np.concatenate(aug_seqs)
    train_labels_aug = np.concatenate(aug_labels)
    train_weights_aug = np.concatenate(aug_weights)
    logger.info(f"After augmentation: {len(train_seqs_aug)} train sequences")

    # Class weights
    weight_sl = len(train_labels) / (2 * n_sl) if n_sl > 0 else 1.0
    weight_tp = len(train_labels) / (2 * n_tp) if n_tp > 0 else 1.0
    class_weights = torch.FloatTensor([weight_sl, weight_tp])

    # Dataloaders
    train_dataset = TensorDataset(
        torch.FloatTensor(train_seqs_aug),
        torch.LongTensor(train_labels_aug),
        torch.FloatTensor(train_weights_aug)
    )
    val_dataset = TensorDataset(torch.FloatTensor(val_seqs), torch.LongTensor(val_labels))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNNDirectionModel(feature_dim=FEATURE_DIM, sequence_length=SEQUENCE_LENGTH, dropout=0.4).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"\nModel: CNNDirectionModel | Params: {n_params:,} | Device: {device}")

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=LABEL_SMOOTHING, reduction='none')
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)

    logger.info(f"LR={LEARNING_RATE} | Batch={BATCH_SIZE} | Patience={PATIENCE} | Augmentation=3x\n")

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

        buy_mask = preds == 1
        buy_precision = labels_arr[buy_mask].mean() * 100 if buy_mask.sum() > 0 else 0

        logger.info(f"E{epoch+1:3d}/{EPOCHS} | "
                     f"T: {train_loss:.4f}/{train_acc:.3f} | "
                     f"V: {val_loss:.4f}/{val_acc:.3f} | "
                     f"BUY={n_buy} SELL={n_sell} | "
                     f"BuyPrec={buy_precision:.0f}% | "
                     f"LR={optimizer.param_groups[0]['lr']:.5f}")

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
            }, MODEL_DIR / 'LINK_direction_model.pt')
            logger.info(f"  >>> BEST (acc={val_acc:.3f}, buyPrec={buy_precision:.0f}%)")
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            logger.info(f"\nEarly stopping at epoch {epoch+1}")
            break

    # Load best
    ckpt = torch.load(MODEL_DIR / 'LINK_direction_model.pt')
    model.load_state_dict(ckpt['model_state_dict'])

    _, _, final_preds, final_labels = validate(model, val_loader, criterion, device)

    logger.info(f"\n{'='*70}")
    logger.info(f"TRAINING COMPLETE | Best epoch: {best_epoch} | Best val loss: {best_val_loss:.4f}")
    logger.info(f"{'='*70}")
    logger.info(f"\n{classification_report(final_labels, final_preds, target_names=['SL', 'TP'])}")

    cm = confusion_matrix(final_labels, final_preds)
    logger.info(f"Confusion: SL[{cm[0][0]},{cm[0][1]}] TP[{cm[1][0]},{cm[1][1]}]")

    # === TEMPERATURE SCALING ===
    logger.info(f"\n{'='*70}")
    logger.info(f"TEMPERATURE SCALING (Confidence Calibration)")
    logger.info(f"{'='*70}")

    model.eval()
    all_logits, all_true_labels = [], []
    with torch.no_grad():
        for X_b, y_b in val_loader:
            logits = model(X_b.to(device))
            all_logits.append(logits.cpu())
            all_true_labels.append(y_b)
    all_logits = torch.cat(all_logits)
    all_true_labels = torch.cat(all_true_labels)

    temperature = nn.Parameter(torch.ones(1) * 1.5)
    temp_optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=100)
    nll_criterion = nn.CrossEntropyLoss()

    def temp_eval():
        temp_optimizer.zero_grad()
        loss = nll_criterion(all_logits / temperature, all_true_labels)
        loss.backward()
        return loss

    temp_optimizer.step(temp_eval)
    optimal_temp = temperature.item()
    logger.info(f"  Optimal temperature: {optimal_temp:.4f}")

    # Save temperature alongside model
    ckpt = torch.load(MODEL_DIR / 'LINK_direction_model.pt')
    ckpt['temperature'] = optimal_temp
    torch.save(ckpt, MODEL_DIR / 'LINK_direction_model.pt')
    logger.info(f"  Temperature saved in model checkpoint")

    # Confidence analysis with calibrated probabilities
    all_confs, all_dirs, all_trues = [], [], []
    with torch.no_grad():
        for X_b, y_b in val_loader:
            logits = model(X_b.to(device))
            calibrated_probs = torch.softmax(logits / optimal_temp, dim=1)
            confidence, direction = torch.max(calibrated_probs, dim=1)
            all_dirs.extend(direction.cpu().numpy())
            all_confs.extend(confidence.cpu().numpy())
            all_trues.extend(y_b.numpy())

    all_dirs = np.array(all_dirs)
    all_confs = np.array(all_confs)
    all_trues = np.array(all_trues)

    logger.info(f"\nCalibrated Confidence Analysis (BUY signals only):")
    for thresh in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        mask = (all_dirs == 1) & (all_confs >= thresh)
        if mask.sum() > 0:
            wr = all_trues[mask].mean() * 100
            logger.info(f"  Conf >= {thresh:.0%}: {mask.sum()} signals, Win Rate: {wr:.1f}%")

    logger.info(f"\nModel size: {(MODEL_DIR / 'LINK_direction_model.pt').stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    train()
