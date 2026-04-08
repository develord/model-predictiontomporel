"""
FIL Meta-Model XGBoost
========================
Takes CNN LONG + SHORT predictions + market features as input.
Decides whether to trust the CNN signal or not.

Pipeline:
1. Load trained CNN LONG + SHORT models
2. Generate predictions on training period (walk-forward)
3. Simulate outcomes (TP/SL hit on historical data)
4. Train XGBoost on: [long_conf, short_conf, long_dir, short_dir, market_features] -> win/lose
5. Save meta-model for use in backtest

Usage:
    python 05_train_meta_xgboost.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import joblib
import json
import logging
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import RobustScaler

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR / 'scripts'))
from direction_prediction_model import CNNDirectionModel, DeepCNNShortModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = BASE_DIR / 'data' / 'cache'
LONG_MODEL_DIR = BASE_DIR / 'models'
SHORT_MODEL_DIR = BASE_DIR / 'models_short'
META_MODEL_DIR = BASE_DIR / 'models_meta'
META_MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Match training script params
SEQUENCE_LENGTH = 30

# Label simulation: ATR-based (matches training labels)
ATR_TP_MULT = 1.5
ATR_SL_MULT = 1.5
FIXED_TP_PCT = 0.012
FIXED_SL_PCT = 0.012
BASE_LOOKAHEAD = 10

# Train meta on historical data, validate on 2025H2 (2026 = true out-of-sample)
META_TRAIN_START = '2020-10-15'
META_TRAIN_END = '2025-06-30'
META_VAL_START = '2025-07-01'
META_VAL_END = '2025-12-31'


def load_cnn_model(model_dir, model_file):
    path = model_dir / model_file
    if not path.exists():
        logger.error(f"Model not found: {path}")
        return None, None, None
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    feat_dim = ckpt.get('feature_dim', 99)
    seq_len = ckpt.get('sequence_length', 30)
    model_type = ckpt.get('model_type', 'cnn')
    # Auto-detect DeepCNNShortModel by state_dict keys
    is_deep = model_type == 'deep_cnn_short' or any('conv3_1' in k or 'conv9_1' in k for k in ckpt['model_state_dict'].keys())
    if is_deep:
        model = DeepCNNShortModel(feature_dim=feat_dim, sequence_length=seq_len, dropout=0.35)
    else:
        model = CNNDirectionModel(feature_dim=feat_dim, sequence_length=seq_len, dropout=0.4)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, seq_len, ckpt


def simulate_long_outcome(df, i, atr_val):
    """Check if LONG trade at index i would hit TP or SL"""
    if i >= len(df) - 2:
        return -1  # Can't evaluate
    entry = df.iloc[i]['close']
    if pd.notna(atr_val) and atr_val > 0:
        tp = entry + atr_val * ATR_TP_MULT
        sl = entry - atr_val * ATR_SL_MULT
    else:
        tp = entry * (1 + FIXED_TP_PCT)
        sl = entry * (1 - FIXED_SL_PCT)

    for j in range(i + 1, min(i + 1 + BASE_LOOKAHEAD, len(df))):
        if df.iloc[j]['high'] >= tp:
            return 1  # Win
        if df.iloc[j]['low'] <= sl:
            return 0  # Lose
    return -1  # No outcome


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
            return 1  # Win
        if df.iloc[j]['high'] >= sl:
            return 0  # Lose
    return -1


def build_meta_features(row, long_conf, long_dir, short_conf, short_dir,
                        long_prob_0, long_prob_1, short_prob_0, short_prob_1):
    """Build feature vector for meta-model from CNN outputs + market context"""
    feat = {
        # CNN outputs (raw)
        'long_conf': long_conf,
        'long_dir': long_dir,
        'short_conf': short_conf,
        'short_dir': short_dir,

        # CNN probability spreads
        'long_prob_spread': long_prob_1 - long_prob_0,   # How decisive is LONG model
        'short_prob_spread': short_prob_1 - short_prob_0,

        # Agreement/conflict signals
        'models_agree_bull': int(long_dir == 1 and short_dir == 0),  # LONG says buy, SHORT says no sell
        'models_agree_bear': int(long_dir == 0 and short_dir == 1),  # LONG says no buy, SHORT says sell
        'models_conflict': int(long_dir == 1 and short_dir == 1),    # Both want to trade (conflict)
        'models_neutral': int(long_dir == 0 and short_dir == 0),     # Neither wants to trade
        'conf_diff': long_conf - short_conf,                         # Who is more confident
        'max_conf': max(long_conf, short_conf),
        'min_conf': min(long_conf, short_conf),
    }

    # Market context features (key ones that help the meta-model decide)
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


def train_meta():
    logger.info(f"\n{'='*70}")
    logger.info(f"FIL META-MODEL XGBOOST TRAINING")
    logger.info(f"{'='*70}\n")

    # Load CNN models
    long_model, long_seq, long_ckpt = load_cnn_model(LONG_MODEL_DIR, 'FIL_direction_model.pt')
    short_model, short_seq, short_ckpt = load_cnn_model(SHORT_MODEL_DIR, 'FIL_short_model.pt')
    if not long_model or not short_model:
        return

    long_temp = long_ckpt.get('temperature', 1.0)
    short_temp = short_ckpt.get('temperature', 1.0)
    logger.info(f"Temperature: LONG={long_temp:.3f}, SHORT={short_temp:.3f}")

    # Load features
    with open(BASE_DIR / 'required_features.json') as f:
        long_feature_cols = json.load(f)
    with open(SHORT_MODEL_DIR / 'short_features.json') as f:
        short_feature_cols = json.load(f)

    long_scaler = joblib.load(LONG_MODEL_DIR / 'feature_scaler.joblib')
    short_scaler = joblib.load(SHORT_MODEL_DIR / 'feature_scaler_short.joblib')

    # Load data
    df = pd.read_csv(DATA_DIR / 'fil_features.csv')
    df['date'] = pd.to_datetime(df['date'])

    # Add bear features for SHORT
    sys.path.insert(0, str(BASE_DIR))
    try:
        from importlib import import_module
        short_train = import_module('03_train_short_model')
        df = short_train.add_bear_features(df)
    except:
        logger.warning("Could not add bear features")

    # Compute ATR for outcome simulation
    import ta
    atr_series = ta.volatility.AverageTrueRange(
        df['high'], df['low'], df['close'], window=14
    ).average_true_range()

    logger.info(f"Data: {len(df)} rows, {df['date'].min().date()} to {df['date'].max().date()}")

    # Prepare scaled features for CNN
    long_raw = df[long_feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    long_feat = np.clip(np.nan_to_num(long_scaler.transform(long_raw), nan=0, posinf=0, neginf=0), -5, 5)

    for c in short_feature_cols:
        if c not in df.columns:
            df[c] = 0
    short_raw = df[short_feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    short_feat = np.clip(np.nan_to_num(short_scaler.transform(short_raw), nan=0, posinf=0, neginf=0), -5, 5)

    seq = max(long_seq, short_seq)

    # Generate predictions + outcomes for each day
    logger.info(f"\nGenerating CNN predictions on full dataset...")
    meta_rows = []

    for i in range(seq, len(df)):
        row = df.iloc[i]
        date = row['date']
        atr_val = atr_series.iloc[i]

        # CNN LONG prediction
        lx = torch.tensor(long_feat[i - long_seq:i], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            l_logits = long_model(lx)
            l_probs = torch.softmax(l_logits / long_temp, dim=1).squeeze()
        l_conf, l_dir = l_probs.max(0)
        l_dir, l_conf = l_dir.item(), l_conf.item()
        l_p0, l_p1 = l_probs[0].item(), l_probs[1].item()

        # CNN SHORT prediction
        sx = torch.tensor(short_feat[i - short_seq:i], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            s_logits = short_model(sx)
            s_probs = torch.softmax(s_logits / short_temp, dim=1).squeeze()
        s_conf, s_dir = s_probs.max(0)
        s_dir, s_conf = s_dir.item(), s_conf.item()
        s_p0, s_p1 = s_probs[0].item(), s_probs[1].item()

        # Simulate outcomes
        long_outcome = simulate_long_outcome(df, i, atr_val)
        short_outcome = simulate_short_outcome(df, i, atr_val)

        # Build meta features
        mf = build_meta_features(row, l_conf, l_dir, s_conf, s_dir, l_p0, l_p1, s_p0, s_p1)
        mf['date'] = date
        mf['long_outcome'] = long_outcome
        mf['short_outcome'] = short_outcome

        meta_rows.append(mf)

    meta_df = pd.DataFrame(meta_rows)
    logger.info(f"Generated {len(meta_df)} prediction rows")

    # === TRAIN META-MODEL FOR LONG ===
    logger.info(f"\n{'='*70}")
    logger.info(f"TRAINING LONG META-MODEL")
    logger.info(f"{'='*70}")

    feature_cols_meta = [c for c in meta_df.columns if c not in
                         ['date', 'long_outcome', 'short_outcome']]

    # Only train on rows where LONG CNN said BUY and outcome is known
    long_train_mask = (
        (meta_df['date'] >= META_TRAIN_START) &
        (meta_df['date'] <= META_TRAIN_END) &
        (meta_df['long_dir'] == 1) &
        (meta_df['long_outcome'] != -1)
    )
    long_val_mask = (
        (meta_df['date'] >= META_VAL_START) &
        (meta_df['date'] <= META_VAL_END) &
        (meta_df['long_dir'] == 1) &
        (meta_df['long_outcome'] != -1)
    )

    lt = meta_df[long_train_mask]
    lv = meta_df[long_val_mask]
    logger.info(f"LONG meta train: {len(lt)} samples | val: {len(lv)} samples")

    if len(lt) > 50:
        X_lt = lt[feature_cols_meta].fillna(0).values
        y_lt = lt['long_outcome'].values
        X_lv = lv[feature_cols_meta].fillna(0).values if len(lv) > 0 else None
        y_lv = lv['long_outcome'].values if len(lv) > 0 else None

        n_win = (y_lt == 1).sum()
        n_lose = (y_lt == 0).sum()
        logger.info(f"  Train dist: Win={n_win} ({n_win/len(y_lt)*100:.0f}%), Lose={n_lose} ({n_lose/len(y_lt)*100:.0f}%)")

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
            early_stopping_rounds=30,
            random_state=42,
            verbosity=0,
        )

        eval_set = [(X_lv, y_lv)] if X_lv is not None and len(X_lv) > 0 else None
        xgb_long.fit(X_lt, y_lt, eval_set=eval_set, verbose=False)

        # Evaluate
        if X_lv is not None and len(X_lv) > 0:
            lv_pred = xgb_long.predict(X_lv)
            lv_prob = xgb_long.predict_proba(X_lv)[:, 1]
            logger.info(f"\n  Val results:")
            logger.info(f"\n{classification_report(y_lv, lv_pred, target_names=['Lose', 'Win'])}")

            # Threshold analysis
            logger.info(f"  Threshold analysis (LONG meta):")
            for t in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
                mask = lv_prob >= t
                if mask.sum() > 0:
                    wr = y_lv[mask].mean() * 100
                    logger.info(f"    Meta >= {t:.0%}: {mask.sum()} signals, WR: {wr:.1f}%")

        # Feature importance
        imp = pd.Series(xgb_long.feature_importances_, index=feature_cols_meta)
        logger.info(f"\n  Top 15 features (LONG meta):")
        for feat, score in imp.nlargest(15).items():
            logger.info(f"    {feat}: {score:.4f}")

        # Save
        joblib.dump(xgb_long, META_MODEL_DIR / 'FIL_meta_long.joblib')
        logger.info(f"\n  Saved: FIL_meta_long.joblib")
    else:
        logger.warning(f"  Not enough LONG training samples ({len(lt)})")

    # === TRAIN META-MODEL FOR SHORT ===
    logger.info(f"\n{'='*70}")
    logger.info(f"TRAINING SHORT META-MODEL")
    logger.info(f"{'='*70}")

    short_train_mask = (
        (meta_df['date'] >= META_TRAIN_START) &
        (meta_df['date'] <= META_TRAIN_END) &
        (meta_df['short_dir'] == 1) &
        (meta_df['short_outcome'] != -1)
    )
    short_val_mask = (
        (meta_df['date'] >= META_VAL_START) &
        (meta_df['date'] <= META_VAL_END) &
        (meta_df['short_dir'] == 1) &
        (meta_df['short_outcome'] != -1)
    )

    st = meta_df[short_train_mask]
    sv = meta_df[short_val_mask]
    logger.info(f"SHORT meta train: {len(st)} samples | val: {len(sv)} samples")

    if len(st) > 50:
        X_st = st[feature_cols_meta].fillna(0).values
        y_st = st['short_outcome'].values
        X_sv = sv[feature_cols_meta].fillna(0).values if len(sv) > 0 else None
        y_sv = sv['short_outcome'].values if len(sv) > 0 else None

        n_win = (y_st == 1).sum()
        n_lose = (y_st == 0).sum()
        logger.info(f"  Train dist: Win={n_win} ({n_win/len(y_st)*100:.0f}%), Lose={n_lose} ({n_lose/len(y_st)*100:.0f}%)")

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
            early_stopping_rounds=30,
            random_state=42,
            verbosity=0,
        )

        eval_set = [(X_sv, y_sv)] if X_sv is not None and len(X_sv) > 0 else None
        xgb_short.fit(X_st, y_st, eval_set=eval_set, verbose=False)

        if X_sv is not None and len(X_sv) > 0:
            sv_pred = xgb_short.predict(X_sv)
            sv_prob = xgb_short.predict_proba(X_sv)[:, 1]
            logger.info(f"\n  Val results:")
            logger.info(f"\n{classification_report(y_sv, sv_pred, target_names=['Lose', 'Win'])}")

            logger.info(f"  Threshold analysis (SHORT meta):")
            for t in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
                mask = sv_prob >= t
                if mask.sum() > 0:
                    wr = y_sv[mask].mean() * 100
                    logger.info(f"    Meta >= {t:.0%}: {mask.sum()} signals, WR: {wr:.1f}%")

        imp = pd.Series(xgb_short.feature_importances_, index=feature_cols_meta)
        logger.info(f"\n  Top 15 features (SHORT meta):")
        for feat, score in imp.nlargest(15).items():
            logger.info(f"    {feat}: {score:.4f}")

        joblib.dump(xgb_short, META_MODEL_DIR / 'FIL_meta_short.joblib')
        logger.info(f"\n  Saved: FIL_meta_short.joblib")
    else:
        logger.warning(f"  Not enough SHORT training samples ({len(st)})")

    # Save meta feature list
    with open(META_MODEL_DIR / 'meta_features.json', 'w') as f:
        json.dump(feature_cols_meta, f, indent=2)

    logger.info(f"\n{'='*70}")
    logger.info(f"META-MODEL TRAINING COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"  Meta features: {len(feature_cols_meta)}")
    logger.info(f"  Files saved in: {META_MODEL_DIR}")


if __name__ == "__main__":
    train_meta()
