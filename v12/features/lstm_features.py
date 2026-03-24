"""
V12 LSTM Feature Generator
===========================
Train an LSTM on price sequences, then use its output as features for XGBoost.

Architecture:
- Input: 20-candle sequences of key indicators (OHLCV normalized + RSI, MACD, ATR, etc.)
- LSTM: 2 layers, 64 hidden units
- Output: P(TP) sigmoid probability

Features generated for XGBoost:
- lstm_proba: raw sigmoid output [0,1]
- lstm_confidence: abs(lstm_proba - 0.5) * 2  (0=uncertain, 1=confident)
- lstm_signal: 1 if lstm_proba > 0.5 else 0
- lstm_agrees_rsi: 1 if lstm_signal matches RSI direction

Anti-leakage strategy:
- LSTM trained on data BEFORE XGBoost's training period
- Expanding window: for each walk-forward period, LSTM sees only prior data
"""

import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, List
import joblib
import json

PROJECT_ROOT = Path(__file__).parent.parent.parent


# ============================================================================
# LSTM MODEL
# ============================================================================

class CryptoLSTM(nn.Module):
    """Simple LSTM binary classifier for TP/SL prediction."""

    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # Use last timestep output
        last_hidden = lstm_out[:, -1, :]
        return self.fc(last_hidden).squeeze(-1)


class SequenceDataset(Dataset):
    """Create sequences for LSTM from time-series data."""

    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = 20):
        self.seq_len = seq_len
        self.sequences = []
        self.labels = []

        for i in range(seq_len, len(X)):
            self.sequences.append(X[i - seq_len:i])
            self.labels.append(y[i])

        self.sequences = np.array(self.sequences, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx]),
            torch.tensor(self.labels[idx])
        )


# ============================================================================
# FEATURE SELECTION FOR LSTM
# ============================================================================

# Key features for LSTM sequences (keep it focused, not all 238+)
LSTM_FEATURES = [
    # Price-derived (normalized as returns)
    'close', 'high', 'low', 'volume',
    # Key indicators from 1d timeframe
    '1d_rsi_14', '1d_macd_line', '1d_macd_histogram',
    '1d_bb_width', '1d_bb_percent',
    '1d_atr_pct', '1d_adx_14',
    '1d_stoch_k', '1d_stoch_d',
    '1d_ema_12', '1d_ema_26',
    '1d_obv', '1d_cmf_20',
    # ATR for volatility context
    'atr_pct_14'
]


def prepare_lstm_features(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """
    Extract and normalize features for LSTM input.
    Returns normalized array and list of feature names used.
    """
    available = [f for f in LSTM_FEATURES if f in df.columns]

    data = df[available].copy()

    # Normalize: price columns as returns, others as z-score
    price_cols = ['close', 'high', 'low']
    for col in price_cols:
        if col in data.columns:
            data[col] = data[col].pct_change()

    # Volume as log ratio
    if 'volume' in data.columns:
        data['volume'] = np.log1p(data['volume']) / np.log1p(data['volume']).rolling(20).mean()

    # Z-score for indicators
    for col in data.columns:
        if col not in price_cols + ['volume']:
            mean = data[col].rolling(50, min_periods=10).mean()
            std = data[col].rolling(50, min_periods=10).std()
            data[col] = (data[col] - mean) / (std + 1e-8)

    # Fill NaN
    data = data.fillna(0)
    data = data.replace([np.inf, -np.inf], 0)

    return data.values, available


# ============================================================================
# TRAINING
# ============================================================================

def train_lstm(
    X_seq: np.ndarray,
    y: np.ndarray,
    seq_len: int = 20,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 0.001,
    verbose: bool = True
) -> CryptoLSTM:
    """Train LSTM on sequence data."""

    dataset = SequenceDataset(X_seq, y, seq_len)
    if len(dataset) < 50:
        raise ValueError(f"Not enough sequences: {len(dataset)}")

    # 90/10 train/val split (temporal)
    val_size = max(int(len(dataset) * 0.1), 10)
    train_size = len(dataset) - val_size

    train_ds = torch.utils.data.Subset(dataset, range(train_size))
    val_ds = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    input_size = X_seq.shape[1]
    model = CryptoLSTM(input_size=input_size)

    # Class weight
    n_pos = y[seq_len:].sum()
    n_neg = len(y[seq_len:]) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # Override criterion to work with sigmoid output
    criterion = nn.BCELoss()

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                pred = model(X_batch)
                val_loss += criterion(pred, y_batch).item()
                val_correct += ((pred > 0.5).float() == y_batch).sum().item()
                val_total += len(y_batch)

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total * 100

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose and (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1:3d}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.1f}%")

        if patience_counter >= 8:
            if verbose:
                print(f"    Early stop at epoch {epoch+1}")
            break

    if best_state:
        model.load_state_dict(best_state)

    return model


# ============================================================================
# INFERENCE - Generate features for XGBoost
# ============================================================================

def generate_lstm_features(
    model: CryptoLSTM,
    X_seq: np.ndarray,
    df: pd.DataFrame,
    seq_len: int = 20
) -> pd.DataFrame:
    """
    Run LSTM inference and generate features for XGBoost.

    Returns DataFrame with columns:
    - lstm_proba, lstm_confidence, lstm_signal, lstm_agrees_rsi
    """
    model.eval()

    # Generate predictions for all valid positions
    probas = np.full(len(X_seq), 0.5)  # Default 0.5 for positions without enough history

    if len(X_seq) > seq_len:
        dataset = SequenceDataset(X_seq, np.zeros(len(X_seq)), seq_len)
        loader = DataLoader(dataset, batch_size=256, shuffle=False)

        all_preds = []
        with torch.no_grad():
            for X_batch, _ in loader:
                pred = model(X_batch)
                all_preds.extend(pred.numpy())

        # Fill in predictions (first seq_len positions don't have predictions)
        probas[seq_len:seq_len + len(all_preds)] = all_preds

    # Build features
    result = pd.DataFrame(index=df.index)
    result['lstm_proba'] = probas[:len(df)]
    result['lstm_confidence'] = np.abs(result['lstm_proba'] - 0.5) * 2
    result['lstm_signal'] = (result['lstm_proba'] > 0.5).astype(int)

    # lstm_agrees_rsi: does LSTM agree with RSI direction?
    if '1d_rsi_14' in df.columns:
        rsi = df['1d_rsi_14'].fillna(50)
        rsi_bullish = (rsi > 50).astype(int)
        result['lstm_agrees_rsi'] = (result['lstm_signal'] == rsi_bullish).astype(int)
    else:
        result['lstm_agrees_rsi'] = 0.5

    return result


# ============================================================================
# FULL PIPELINE
# ============================================================================

def build_lstm_features_for_crypto(
    crypto: str,
    lstm_train_end: str,
    seq_len: int = 20,
    epochs: int = 30,
    verbose: bool = True
) -> Tuple[pd.DataFrame, CryptoLSTM]:
    """
    Full pipeline: train LSTM and generate features for XGBoost.

    Args:
        crypto: 'btc', 'eth', 'sol'
        lstm_train_end: cutoff date for LSTM training (e.g., '2024-01-01')
        seq_len: LSTM sequence length
        epochs: training epochs

    Returns:
        DataFrame with LSTM features for ALL data points
        Trained LSTM model
    """
    from v12.features.dynamic_labels import calculate_atr_series

    # Load data
    cache_file = PROJECT_ROOT / 'data' / 'cache' / f'{crypto}_multi_tf_merged.csv'
    df = pd.read_csv(cache_file, index_col=0, parse_dates=True)

    # Add ATR if not present
    if 'atr_pct_14' not in df.columns:
        df['atr_pct_14'] = calculate_atr_series(df, 14)

    if verbose:
        print(f"\n  LSTM for {crypto.upper()}: {len(df)} rows, train cutoff={lstm_train_end}")

    # Prepare features
    X_norm, feat_names = prepare_lstm_features(df)
    if verbose:
        print(f"  LSTM input: {len(feat_names)} features, seq_len={seq_len}")

    # Prepare labels (V11 fixed triple barrier)
    y_full = df['triple_barrier_label'].fillna(0).values
    y_binary = (y_full == 1).astype(float)

    # Split for LSTM training
    train_mask = df.index < lstm_train_end
    X_train = X_norm[train_mask]
    y_train = y_binary[train_mask]

    n_tp = y_train.sum()
    n_sl = len(y_train) - n_tp
    if verbose:
        print(f"  LSTM train: {len(X_train)} samples (TP={int(n_tp)}, SL={int(n_sl)})")

    # Train LSTM
    if verbose:
        print(f"  Training LSTM ({epochs} epochs)...")

    model = train_lstm(X_train, y_train, seq_len=seq_len, epochs=epochs, verbose=verbose)

    # Generate features for ALL data
    if verbose:
        print(f"  Generating LSTM features for full dataset...")

    lstm_features = generate_lstm_features(model, X_norm, df, seq_len)

    # Stats
    if verbose:
        test_mask = df.index >= lstm_train_end
        test_proba = lstm_features.loc[test_mask, 'lstm_proba']
        test_signal = lstm_features.loc[test_mask, 'lstm_signal']
        test_conf = lstm_features.loc[test_mask, 'lstm_confidence']
        print(f"  LSTM test stats:")
        print(f"    P(TP) mean={test_proba.mean():.4f} std={test_proba.std():.4f}")
        print(f"    Signal: {test_signal.sum():.0f} BUY / {(1-test_signal).sum():.0f} SKIP")
        print(f"    Confidence mean={test_conf.mean():.4f}")
        agrees = lstm_features.loc[test_mask, 'lstm_agrees_rsi']
        print(f"    Agrees with RSI: {agrees.mean()*100:.1f}%")

    return lstm_features, model


def save_lstm_model(model: CryptoLSTM, crypto: str, metadata: dict = None):
    """Save LSTM model and metadata."""
    models_dir = Path(__file__).parent.parent / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), models_dir / f'{crypto}_lstm.pt')

    if metadata:
        with open(models_dir / f'{crypto}_lstm_meta.json', 'w') as f:
            json.dump(metadata, f, indent=2)


def load_lstm_model(crypto: str, input_size: int) -> CryptoLSTM:
    """Load saved LSTM model."""
    models_dir = Path(__file__).parent.parent / 'models'
    model = CryptoLSTM(input_size=input_size)
    model.load_state_dict(torch.load(models_dir / f'{crypto}_lstm.pt', weights_only=True))
    model.eval()
    return model
