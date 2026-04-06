"""
ETH Production Training Script
================================
Retrains ALL models on the FULL dataset (no train/val split) for production deployment.

Models trained:
1. LONG CNN  (CNNDirectionModel) -> models/ETH_direction_model.pt
2. SHORT CNN (CNNDirectionModel) -> models_short/ETH_short_model.pt
3. Meta XGBoost LONG             -> models_meta/ETH_meta_long.joblib
4. Meta XGBoost SHORT            -> models_meta/ETH_meta_short.joblib

Temperature values are HARDCODED from previous calibrated runs (no val set to optimize on).

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
from xgboost import XGBClassifier
import ta as ta_lib

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR / 'scripts'))

from direction_prediction_model import CNNDirectionModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = BASE_DIR / 'data' / 'cache'
LONG_MODEL_DIR = BASE_DIR / 'models'
SHORT_MODEL_DIR = BASE_DIR / 'models_short'
META_MODEL_DIR = BASE_DIR / 'models_meta'
LONG_MODEL_DIR.mkdir(parents=True, exist_ok=True)
SHORT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
META_MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Shared constants
# ============================================================================
SEQUENCE_LENGTH = 30
BATCH_SIZE = 64
GRAD_CLIP = 0.5

# ATR labeling (shared for LONG and SHORT)
ATR_TP_MULT = 1.5
ATR_SL_MULT = 1.5
FIXED_TP_PCT = 0.012
FIXED_SL_PCT = 0.012
BASE_LOOKAHEAD = 10
ATR_LOOKAHEAD_MULT = 0.7

# LONG CNN params
LONG_FIXED_EPOCHS = 5
LONG_LR = 0.0015
LONG_NOISE_STD = 0.015
LONG_LABEL_SMOOTHING = 0.1
LONG_AUGMENT_COPIES = 2   # 3x total (original + 2 copies)
LONG_TEMPERATURE = 1.295  # Hardcoded from calibrated run

# SHORT CNN params
SHORT_FIXED_EPOCHS = 65
SHORT_LR = 0.0005
SHORT_NOISE_STD = 0.02
SHORT_LABEL_SMOOTHING = 0.05
SHORT_AUGMENT_COPIES = 3  # 4x total (original + 3 copies)
SHORT_TEMPERATURE = 1.807  # Hardcoded from calibrated run


# ============================================================================
# Label creation functions
# ============================================================================

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
    """ATR-adaptive SHORT labels (price drops = TP, price rises = SL)."""
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
        tp = entry - tp_dist  # SHORT TP = price drops
        sl = entry + sl_dist  # SHORT SL = price rises
        med = median_atr.iloc[i] if pd.notna(median_atr.iloc[i]) and median_atr.iloc[i] > 0 else cur_atr
        if pd.notna(med) and med > 0:
            vol_ratio = cur_atr / med
            lookahead = int(BASE_LOOKAHEAD * max(0.5, min(2.0, vol_ratio * ATR_LOOKAHEAD_MULT + 0.3)))
        else:
            lookahead = BASE_LOOKAHEAD
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


# ============================================================================
# Bear features (same as 03_train_short_model.py)
# ============================================================================

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


# ============================================================================
# Training helper
# ============================================================================

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


# ============================================================================
# Meta-model helpers (from 05_train_meta_xgboost.py)
# ============================================================================

def simulate_long_outcome(df, i, atr_val):
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
    """Build feature vector for meta-model from CNN outputs + market context."""
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


# ============================================================================
# STEP 1: Train LONG CNN on full dataset
# ============================================================================

def train_long_cnn(df, long_feature_cols, device):
    logger.info(f"\n{'=' * 70}")
    logger.info(f"STEP 1: LONG CNN - Production Training (ALL data, {LONG_FIXED_EPOCHS} epochs)")
    logger.info(f"{'=' * 70}")

    FEATURE_DIM = len(long_feature_cols)
    logger.info(f"Features: {FEATURE_DIM} | Seq: {SEQUENCE_LENGTH} | LR: {LONG_LR}")

    # Create LONG labels
    df_long = df.copy()
    df_long['label'] = create_long_labels(df_long)

    # Scale ALL data
    X_raw = df_long[long_feature_cols].fillna(0).values.astype(np.float32)
    y_raw = df_long['label'].values

    scaler = RobustScaler()
    X_scaled = np.nan_to_num(scaler.fit_transform(X_raw), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    X_scaled = np.clip(X_scaled, -5, 5)

    joblib.dump(scaler, LONG_MODEL_DIR / 'feature_scaler.joblib')
    logger.info(f"Saved feature_scaler.joblib")

    # Build sequences with bear weighting
    bear_flag = df_long['is_strong_bear'].values if 'is_strong_bear' in df_long.columns else np.zeros(len(df_long))

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
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Model
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

    # Train for fixed epochs (no early stopping - no val set)
    for epoch in range(LONG_FIXED_EPOCHS):
        train_loss, train_acc = train_epoch_weighted(
            model, train_loader, criterion, optimizer, device, GRAD_CLIP
        )
        scheduler.step(epoch)
        logger.info(f"  E{epoch + 1}/{LONG_FIXED_EPOCHS} | Loss: {train_loss:.4f} | Acc: {train_acc:.3f} | "
                     f"LR: {optimizer.param_groups[0]['lr']:.6f}")

    # Save with hardcoded temperature
    torch.save({
        'model_state_dict': model.state_dict(),
        'feature_dim': FEATURE_DIM,
        'sequence_length': SEQUENCE_LENGTH,
        'model_type': 'cnn',
        'temperature': LONG_TEMPERATURE,
        'epoch': LONG_FIXED_EPOCHS,
        'production': True,
    }, LONG_MODEL_DIR / 'ETH_direction_model.pt')

    size_kb = (LONG_MODEL_DIR / 'ETH_direction_model.pt').stat().st_size / 1024
    logger.info(f"Saved: models/ETH_direction_model.pt ({size_kb:.0f} KB)")
    logger.info(f"Hardcoded temperature: {LONG_TEMPERATURE}")

    return model, scaler


# ============================================================================
# STEP 2: Train SHORT CNN on full dataset
# ============================================================================

def train_short_cnn(df, base_feature_cols, device):
    logger.info(f"\n{'=' * 70}")
    logger.info(f"STEP 2: SHORT CNN - Production Training (ALL data, {SHORT_FIXED_EPOCHS} epochs)")
    logger.info(f"{'=' * 70}")

    # Add bear features
    df_short = df.copy()
    df_short = add_bear_features(df_short)

    bear_features = [
        'price_above_sma10_pct', 'price_above_sma20_pct', 'price_above_sma50_pct',
        'roc_5', 'roc_10', 'roc_deceleration',
        'vol_price_divergence', 'consec_red', 'high_rejection',
        'dist_from_high_20', 'dist_from_high_50', 'vol_expansion',
    ]
    if 'rsi_price_divergence' in df_short.columns:
        bear_features.extend(['rsi_slope_5', 'price_slope_5', 'rsi_price_divergence'])

    short_feature_cols = base_feature_cols + [f for f in bear_features if f in df_short.columns]
    FEATURE_DIM = len(short_feature_cols)
    logger.info(f"Features: {FEATURE_DIM} ({len(base_feature_cols)} base + {FEATURE_DIM - len(base_feature_cols)} bear)")

    # Create SHORT labels
    df_short['label'] = create_short_labels(df_short)
    n1 = (df_short['label'] == 1).sum()
    n0 = (df_short['label'] == 0).sum()
    logger.info(f"SHORT labels: Profitable={n1} | Not={n0} | Ratio: {n1 / (n0 + 1):.2f}")

    # Scale ALL data
    X_raw = df_short[short_feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    y_raw = df_short['label'].values

    scaler = RobustScaler()
    X_scaled = np.clip(
        np.nan_to_num(scaler.fit_transform(X_raw), nan=0, posinf=0, neginf=0), -5, 5
    ).astype(np.float32)

    joblib.dump(scaler, SHORT_MODEL_DIR / 'feature_scaler_short.joblib')

    # Build sequences
    train_seqs, train_labels = [], []
    for i in range(SEQUENCE_LENGTH, len(X_scaled)):
        if y_raw[i] != -1:
            train_seqs.append(X_scaled[i - SEQUENCE_LENGTH:i])
            train_labels.append(y_raw[i])
    train_seqs = np.array(train_seqs)
    train_labels = np.array(train_labels)

    n_0 = (train_labels == 0).sum()
    n_1 = (train_labels == 1).sum()
    logger.info(f"Sequences: {len(train_seqs)} | No={n_0} Short={n_1}")

    # Augmentation (4x total)
    aug_seqs, aug_labels = [train_seqs], [train_labels]
    for _ in range(SHORT_AUGMENT_COPIES):
        aug_seqs.append(train_seqs + np.random.normal(0, SHORT_NOISE_STD, train_seqs.shape).astype(np.float32))
        aug_labels.append(train_labels)
    train_seqs_aug = np.concatenate(aug_seqs)
    train_labels_aug = np.concatenate(aug_labels)
    logger.info(f"After augmentation: {len(train_seqs_aug)} sequences (4x)")

    # Class weights
    w0 = len(train_labels) / (2 * n_0) if n_0 > 0 else 1.0
    w1 = len(train_labels) / (2 * n_1) if n_1 > 0 else 1.0

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(train_seqs_aug), torch.LongTensor(train_labels_aug)),
        batch_size=BATCH_SIZE, shuffle=True
    )

    # Model
    model = CNNDirectionModel(feature_dim=FEATURE_DIM, sequence_length=SEQUENCE_LENGTH, dropout=0.4).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: CNNDirectionModel (SHORT) | Params: {n_params:,} | Device: {device}")

    criterion = nn.CrossEntropyLoss(
        weight=torch.FloatTensor([w0, w1]).to(device),
        label_smoothing=SHORT_LABEL_SMOOTHING
    )
    optimizer = optim.AdamW(model.parameters(), lr=SHORT_LR, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

    # Train for fixed epochs
    for epoch in range(SHORT_FIXED_EPOCHS):
        model.train()
        epoch_loss, epoch_correct, epoch_total = 0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            epoch_loss += loss.item()
            epoch_correct += (model(xb).argmax(1) == yb).sum().item()
            epoch_total += yb.size(0)
        scheduler.step(epoch)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            acc = epoch_correct / epoch_total if epoch_total > 0 else 0
            logger.info(f"  E{epoch + 1}/{SHORT_FIXED_EPOCHS} | Loss: {epoch_loss / len(train_loader):.4f} | "
                         f"Acc: {acc:.3f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

    # Save with hardcoded temperature
    torch.save({
        'model_state_dict': model.state_dict(),
        'feature_dim': FEATURE_DIM,
        'sequence_length': SEQUENCE_LENGTH,
        'model_type': 'cnn',
        'temperature': SHORT_TEMPERATURE,
        'short_atr_tp_mult': ATR_TP_MULT,
        'short_atr_sl_mult': ATR_SL_MULT,
        'epoch': SHORT_FIXED_EPOCHS,
        'production': True,
    }, SHORT_MODEL_DIR / 'ETH_short_model.pt')

    # Save feature list
    with open(SHORT_MODEL_DIR / 'short_features.json', 'w') as f:
        json.dump(short_feature_cols, f)

    size_kb = (SHORT_MODEL_DIR / 'ETH_short_model.pt').stat().st_size / 1024
    logger.info(f"Saved: models_short/ETH_short_model.pt ({size_kb:.0f} KB)")
    logger.info(f"Saved: models_short/short_features.json ({FEATURE_DIM} features)")
    logger.info(f"Hardcoded temperature: {SHORT_TEMPERATURE}")

    return model, scaler, short_feature_cols


# ============================================================================
# STEP 3: Train Meta XGBoost (LONG + SHORT) on full dataset
# ============================================================================

def train_meta_xgboost(df, long_model, long_scaler, long_feature_cols,
                       short_model, short_scaler, short_feature_cols, device):
    logger.info(f"\n{'=' * 70}")
    logger.info(f"STEP 3: META XGBOOST - Production Training (ALL data, no early stopping)")
    logger.info(f"{'=' * 70}")

    # Add bear features to df for SHORT
    df_meta = df.copy()
    df_meta = add_bear_features(df_meta)

    # Compute ATR for outcome simulation
    atr_series = ta_lib.volatility.AverageTrueRange(
        df_meta['high'], df_meta['low'], df_meta['close'], window=14
    ).average_true_range()

    # Prepare scaled features for CNN inference
    long_raw = df_meta[long_feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    long_feat = np.clip(
        np.nan_to_num(long_scaler.transform(long_raw), nan=0, posinf=0, neginf=0), -5, 5
    )

    for c in short_feature_cols:
        if c not in df_meta.columns:
            df_meta[c] = 0
    short_raw = df_meta[short_feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    short_feat = np.clip(
        np.nan_to_num(short_scaler.transform(short_raw), nan=0, posinf=0, neginf=0), -5, 5
    )

    # Generate CNN predictions on full dataset
    logger.info("Generating CNN predictions on full dataset...")
    long_model.eval()
    short_model.eval()
    meta_rows = []

    for i in range(SEQUENCE_LENGTH, len(df_meta)):
        row = df_meta.iloc[i]
        date = row['date']
        atr_val = atr_series.iloc[i]

        # LONG prediction
        lx = torch.tensor(long_feat[i - SEQUENCE_LENGTH:i], dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            l_logits = long_model(lx)
            l_probs = torch.softmax(l_logits / LONG_TEMPERATURE, dim=1).squeeze()
        l_conf, l_dir = l_probs.max(0)
        l_dir, l_conf = l_dir.item(), l_conf.item()
        l_p0, l_p1 = l_probs[0].item(), l_probs[1].item()

        # SHORT prediction
        sx = torch.tensor(short_feat[i - SEQUENCE_LENGTH:i], dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            s_logits = short_model(sx)
            s_probs = torch.softmax(s_logits / SHORT_TEMPERATURE, dim=1).squeeze()
        s_conf, s_dir = s_probs.max(0)
        s_dir, s_conf = s_dir.item(), s_conf.item()
        s_p0, s_p1 = s_probs[0].item(), s_probs[1].item()

        # Simulate outcomes
        long_outcome = simulate_long_outcome(df_meta, i, atr_val)
        short_outcome = simulate_short_outcome(df_meta, i, atr_val)

        mf = build_meta_features(row, l_conf, l_dir, s_conf, s_dir, l_p0, l_p1, s_p0, s_p1)
        mf['date'] = date
        mf['long_outcome'] = long_outcome
        mf['short_outcome'] = short_outcome
        meta_rows.append(mf)

    meta_df = pd.DataFrame(meta_rows)
    logger.info(f"Generated {len(meta_df)} meta prediction rows")

    feature_cols_meta = [c for c in meta_df.columns if c not in ['date', 'long_outcome', 'short_outcome']]

    # === LONG META ===
    logger.info(f"\n--- LONG META-MODEL ---")
    long_mask = (meta_df['long_dir'] == 1) & (meta_df['long_outcome'] != -1)
    lt = meta_df[long_mask]
    logger.info(f"LONG meta samples: {len(lt)}")

    if len(lt) > 50:
        X_lt = lt[feature_cols_meta].fillna(0).values
        y_lt = lt['long_outcome'].values
        n_win = (y_lt == 1).sum()
        n_lose = (y_lt == 0).sum()
        logger.info(f"  Win={n_win} ({n_win / len(y_lt) * 100:.0f}%), Lose={n_lose} ({n_lose / len(y_lt) * 100:.0f}%)")

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
        xgb_long.fit(X_lt, y_lt)

        # Feature importance
        imp = pd.Series(xgb_long.feature_importances_, index=feature_cols_meta)
        logger.info(f"  Top 10 features (LONG meta):")
        for feat, score in imp.nlargest(10).items():
            logger.info(f"    {feat}: {score:.4f}")

        joblib.dump(xgb_long, META_MODEL_DIR / 'ETH_meta_long.joblib')
        logger.info(f"  Saved: models_meta/ETH_meta_long.joblib")
    else:
        logger.warning(f"  Not enough LONG samples ({len(lt)}), skipping meta LONG")

    # === SHORT META ===
    logger.info(f"\n--- SHORT META-MODEL ---")
    short_mask = (meta_df['short_dir'] == 1) & (meta_df['short_outcome'] != -1)
    st = meta_df[short_mask]
    logger.info(f"SHORT meta samples: {len(st)}")

    if len(st) > 50:
        X_st = st[feature_cols_meta].fillna(0).values
        y_st = st['short_outcome'].values
        n_win = (y_st == 1).sum()
        n_lose = (y_st == 0).sum()
        logger.info(f"  Win={n_win} ({n_win / len(y_st) * 100:.0f}%), Lose={n_lose} ({n_lose / len(y_st) * 100:.0f}%)")

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

        imp = pd.Series(xgb_short.feature_importances_, index=feature_cols_meta)
        logger.info(f"  Top 10 features (SHORT meta):")
        for feat, score in imp.nlargest(10).items():
            logger.info(f"    {feat}: {score:.4f}")

        joblib.dump(xgb_short, META_MODEL_DIR / 'ETH_meta_short.joblib')
        logger.info(f"  Saved: models_meta/ETH_meta_short.joblib")
    else:
        logger.warning(f"  Not enough SHORT samples ({len(st)}), skipping meta SHORT")

    # Save meta feature list
    with open(META_MODEL_DIR / 'meta_features.json', 'w') as f:
        json.dump(feature_cols_meta, f, indent=2)
    logger.info(f"\nSaved: models_meta/meta_features.json ({len(feature_cols_meta)} features)")


# ============================================================================
# MAIN
# ============================================================================

def main():
    logger.info(f"\n{'#' * 70}")
    logger.info(f"# ETH PRODUCTION TRAINING - ALL MODELS ON FULL DATASET")
    logger.info(f"{'#' * 70}\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    # Load data
    df = pd.read_csv(DATA_DIR / 'eth_features.csv')
    df['date'] = pd.to_datetime(df['date'])
    logger.info(f"Data: {len(df)} rows, {df['date'].min().date()} to {df['date'].max().date()}")

    # Load base feature list
    with open(BASE_DIR / 'required_features.json') as f:
        base_feature_cols = json.load(f)
    logger.info(f"Base features: {len(base_feature_cols)}")

    # STEP 1: LONG CNN
    long_model, long_scaler = train_long_cnn(df, base_feature_cols, device)

    # STEP 2: SHORT CNN
    short_model, short_scaler, short_feature_cols = train_short_cnn(df, base_feature_cols, device)

    # STEP 3: Meta XGBoost
    train_meta_xgboost(
        df, long_model, long_scaler, base_feature_cols,
        short_model, short_scaler, short_feature_cols, device
    )

    # Summary
    logger.info(f"\n{'#' * 70}")
    logger.info(f"# PRODUCTION TRAINING COMPLETE")
    logger.info(f"{'#' * 70}")
    logger.info(f"")
    logger.info(f"Files produced:")
    logger.info(f"  models/ETH_direction_model.pt        (LONG CNN, temp={LONG_TEMPERATURE})")
    logger.info(f"  models/feature_scaler.joblib          (LONG scaler)")
    logger.info(f"  models_short/ETH_short_model.pt       (SHORT CNN, temp={SHORT_TEMPERATURE})")
    logger.info(f"  models_short/feature_scaler_short.joblib (SHORT scaler)")
    logger.info(f"  models_short/short_features.json      (SHORT feature list)")
    logger.info(f"  models_meta/ETH_meta_long.joblib      (Meta XGB LONG)")
    logger.info(f"  models_meta/ETH_meta_short.joblib     (Meta XGB SHORT)")
    logger.info(f"  models_meta/meta_features.json        (Meta feature list)")
    logger.info(f"")
    logger.info(f"All models trained on FULL dataset. Ready for production deployment.")


if __name__ == "__main__":
    main()
