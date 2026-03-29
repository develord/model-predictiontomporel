"""
BTC XGBoost V12 - ENHANCED MULTI-TIMEFRAME FEATURES
===================================================

TEST 2: Training NEW BTC model with advanced multi-timeframe feature engineering

Enhancements beyond V11 (237 features):
1. Timeframe Alignment Indicators (when all TFs show same signal)
2. Cross-Timeframe Momentum Correlations
3. Advanced Volume Profile across TFs
4. Support/Resistance Multi-TF Validation
5. Volatility Regime Clustering
6. Price Action Coherence across TFs
7. Momentum Divergence Detection
8. Market Regime Transitions

Target: Beat V11 Baseline (+22.56% ROI) with better feature engineering

Date: 2026-03-29
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
import logging
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, roc_auc_score
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
CRYPTO = 'BTC'
BASE_DIR = Path(__file__).parent.parent
DATA_FILE = BASE_DIR / 'data' / 'cache' / 'btc_multi_tf_merged.csv'
MODEL_DIR = BASE_DIR / 'models'
MODEL_NAME = 'btc_v12_enhanced.joblib'

# Label configuration (TP/SL system)
TP_PCT = 0.015  # 1.5% target
SL_PCT = 0.0075  # 0.75% stop-loss
FORWARD_DAYS = 5  # Look forward 5 days

# Training configuration
TEST_SIZE = 0.15
RANDOM_STATE = 42


def create_advanced_multi_tf_features(df):
    """
    Create advanced multi-timeframe features beyond the existing 284 features.

    Works with the actual data structure (1d_*, 4h_*, 1w_* prefixed features).
    Returns enriched dataframe with 50+ NEW features focused on TF coherence.
    """
    logger.info("Creating advanced multi-timeframe features...")

    df = df.copy()

    # ====================================================================
    # 1. TIMEFRAME ALIGNMENT INDICATORS
    # ====================================================================
    logger.info("  [1/6] Timeframe alignment indicators...")

    # RSI alignment (all TFs show same overbought/oversold)
    df['rsi_all_oversold'] = ((df['1d_rsi_14'] < 30) &
                               (df['4h_rsi_14'] < 30) &
                               (df['1w_rsi_14'] < 30)).astype(int)

    df['rsi_all_overbought'] = ((df['1d_rsi_14'] > 70) &
                                 (df['4h_rsi_14'] > 70) &
                                 (df['1w_rsi_14'] > 70)).astype(int)

    df['rsi_alignment_score'] = (
        (df['1d_rsi_14'] > 50).astype(int) +
        (df['4h_rsi_14'] > 50).astype(int) +
        (df['1w_rsi_14'] > 50).astype(int)
    ) / 3.0  # 0-1 score

    # Momentum alignment
    df['momentum_all_positive'] = ((df['1d_momentum_5'] > 0) &
                                    (df['4h_momentum_5'] > 0) &
                                    (df['1w_momentum_5'] > 0)).astype(int)

    df['momentum_all_negative'] = ((df['1d_momentum_5'] < 0) &
                                    (df['4h_momentum_5'] < 0) &
                                    (df['1w_momentum_5'] < 0)).astype(int)

    # Trend alignment using close price relative to EMA
    df['trend_all_bullish'] = ((df['close'] > df['1d_ema_50']) &
                                (df['close'] > df['4h_ema_50']) &
                                (df['close'] > df['1w_ema_50'])).astype(int)

    df['trend_all_bearish'] = ((df['close'] < df['1d_ema_50']) &
                                (df['close'] < df['4h_ema_50']) &
                                (df['close'] < df['1w_ema_50'])).astype(int)

    # ====================================================================
    # 2. CROSS-TIMEFRAME MOMENTUM CORRELATIONS
    # ====================================================================
    logger.info("  [2/6] Cross-timeframe momentum correlations...")

    # Rolling correlation between TF momentums
    df['momentum_1d_4h_corr'] = df['1d_momentum_5'].rolling(20).corr(df['4h_momentum_5'])
    df['momentum_1d_1w_corr'] = df['1d_momentum_5'].rolling(20).corr(df['1w_momentum_5'])
    df['momentum_4h_1w_corr'] = df['4h_momentum_5'].rolling(20).corr(df['1w_momentum_5'])

    # Momentum divergence (when TFs disagree)
    df['momentum_divergence_1d_4h'] = np.abs(
        np.sign(df['1d_momentum_5']) - np.sign(df['4h_momentum_5'])
    )
    df['momentum_divergence_1d_1w'] = np.abs(
        np.sign(df['1d_momentum_5']) - np.sign(df['1w_momentum_5'])
    )

    # Momentum strength ratio
    df['momentum_ratio_1d_4h'] = df['1d_momentum_5'] / (df['4h_momentum_5'].abs() + 1e-8)
    df['momentum_ratio_1d_1w'] = df['1d_momentum_5'] / (df['1w_momentum_5'].abs() + 1e-8)

    # ====================================================================
    # 3. VOLUME TREND ALIGNMENT
    # ====================================================================
    logger.info("  [3/6] Volume trend alignment...")

    # Volume trend alignment
    df['volume_trend_all_rising'] = (
        (df['1d_volume_trend_7'] > 0) &
        (df['4h_volume_trend_7'] > 0) &
        (df['1w_volume_trend_7'] > 0)
    ).astype(int)

    df['volume_trend_all_falling'] = (
        (df['1d_volume_trend_7'] < 0) &
        (df['4h_volume_trend_7'] < 0) &
        (df['1w_volume_trend_7'] < 0)
    ).astype(int)

    # Volume ratio alignment (all TFs above average)
    df['volume_ratio_all_strong'] = (
        (df['1d_volume_ratio_7'] > 1.0) &
        (df['4h_volume_ratio_7'] > 1.0) &
        (df['1w_volume_ratio_7'] > 1.0)
    ).astype(int)

    # ====================================================================
    # 4. VOLATILITY REGIME CLUSTERING
    # ====================================================================
    logger.info("  [4/6] Volatility regime clustering...")

    # Volatility alignment (all TFs in same regime)
    df['volatility_all_low'] = (
        (df['1d_hist_vol_20'] < df['1d_hist_vol_20'].quantile(0.33)) &
        (df['4h_hist_vol_20'] < df['4h_hist_vol_20'].quantile(0.33)) &
        (df['1w_hist_vol_20'] < df['1w_hist_vol_20'].quantile(0.33))
    ).astype(int)

    df['volatility_all_high'] = (
        (df['1d_hist_vol_20'] > df['1d_hist_vol_20'].quantile(0.67)) &
        (df['4h_hist_vol_20'] > df['4h_hist_vol_20'].quantile(0.67)) &
        (df['1w_hist_vol_20'] > df['1w_hist_vol_20'].quantile(0.67))
    ).astype(int)

    # Volatility expansion/contraction
    df['volatility_expanding'] = (
        (df['1d_hist_vol_20'] > df['1d_hist_vol_20'].shift(5)) &
        (df['4h_hist_vol_20'] > df['4h_hist_vol_20'].shift(5)) &
        (df['1w_hist_vol_20'] > df['1w_hist_vol_20'].shift(5))
    ).astype(int)

    df['volatility_contracting'] = (
        (df['1d_hist_vol_20'] < df['1d_hist_vol_20'].shift(5)) &
        (df['4h_hist_vol_20'] < df['4h_hist_vol_20'].shift(5)) &
        (df['1w_hist_vol_20'] < df['1w_hist_vol_20'].shift(5))
    ).astype(int)

    # ====================================================================
    # 5. PRICE ACTION COHERENCE
    # ====================================================================
    logger.info("  [5/6] Price action coherence...")

    # RSI correlation across timeframes
    df['rsi_1d_4h_corr'] = df['1d_rsi_14'].rolling(20).corr(df['4h_rsi_14'])
    df['rsi_1d_1w_corr'] = df['1d_rsi_14'].rolling(20).corr(df['1w_rsi_14'])

    # MACD histogram correlation
    df['macd_1d_4h_corr'] = df['1d_macd_histogram'].rolling(20).corr(df['4h_macd_histogram'])
    df['macd_1d_1w_corr'] = df['1d_macd_histogram'].rolling(20).corr(df['1w_macd_histogram'])

    # ====================================================================
    # 6. MARKET REGIME TRANSITIONS
    # ====================================================================
    logger.info("  [6/6] Market regime transitions...")

    # ADX regime (trend strength)
    df['adx_all_strong_trend'] = (
        (df['1d_adx_14'] > 25) &
        (df['4h_adx_14'] > 25) &
        (df['1w_adx_14'] > 25)
    ).astype(int)

    df['adx_all_weak_trend'] = (
        (df['1d_adx_14'] < 20) &
        (df['4h_adx_14'] < 20) &
        (df['1w_adx_14'] < 20)
    ).astype(int)

    # ADX increasing (trend strengthening)
    df['adx_strengthening'] = (
        (df['1d_adx_14'] > df['1d_adx_14'].shift(5)) &
        (df['4h_adx_14'] > df['4h_adx_14'].shift(5))
    ).astype(int)

    # BB width correlation (volatility sync)
    df['bb_width_1d_4h_corr'] = df['1d_bb_width'].rolling(20).corr(df['4h_bb_width'])
    df['bb_width_1d_1w_corr'] = df['1d_bb_width'].rolling(20).corr(df['1w_bb_width'])

    # Fill NaN values
    df = df.fillna(0)
    df = df.replace([np.inf, -np.inf], 0)

    # Count new features
    initial_feature_count = 284  # Known from data inspection
    new_feature_count = len(df.columns) - initial_feature_count
    logger.info(f"  [OK] Created {new_feature_count} NEW advanced features")
    logger.info(f"  [OK] Total features: {len(df.columns)}")

    return df


def create_tp_sl_labels(df, tp_pct=TP_PCT, sl_pct=SL_PCT, forward_days=FORWARD_DAYS):
    """
    Create TP/SL labels: 1 if TP hit first, 0 if SL hit first.
    """
    logger.info(f"Creating TP/SL labels (TP: {tp_pct*100:.1f}%, SL: {sl_pct*100:.2f}%)...")

    labels = []

    for i in range(len(df)):
        if i >= len(df) - forward_days:
            labels.append(np.nan)
            continue

        entry_price = df['close'].iloc[i]
        tp_price = entry_price * (1 + tp_pct)
        sl_price = entry_price * (1 - sl_pct)

        # Check forward candles
        hit_tp = False
        hit_sl = False

        for j in range(i + 1, min(i + forward_days + 1, len(df))):
            high = df['high'].iloc[j]
            low = df['low'].iloc[j]

            if high >= tp_price:
                hit_tp = True
                break
            if low <= sl_price:
                hit_sl = True
                break

        # Label: 1 = TP, 0 = SL, NaN = neither
        if hit_tp:
            labels.append(1)
        elif hit_sl:
            labels.append(0)
        else:
            labels.append(np.nan)

    df['label'] = labels

    # Remove NaN labels
    df_labeled = df.dropna(subset=['label']).copy()

    logger.info(f"  Labels created: {len(df_labeled)} samples")
    logger.info(f"  TP (1): {(df_labeled['label'] == 1).sum()} ({(df_labeled['label'] == 1).mean()*100:.1f}%)")
    logger.info(f"  SL (0): {(df_labeled['label'] == 0).sum()} ({(df_labeled['label'] == 0).mean()*100:.1f}%)")

    return df_labeled


def train_xgboost(X_train, y_train, X_test, y_test):
    """Train XGBoost with optimized hyperparameters."""
    logger.info("Training XGBoost V12...")

    # Calculate scale_pos_weight for class imbalance
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    params = {
        'objective': 'binary:logistic',
        'eval_metric': ['auc', 'logloss'],
        'max_depth': 7,
        'learning_rate': 0.03,
        'n_estimators': 500,
        'min_child_weight': 3,
        'gamma': 0.2,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': scale_pos_weight,
        'reg_alpha': 0.1,
        'reg_lambda': 1.5,
        'random_state': RANDOM_STATE,
        'tree_method': 'hist',
        'early_stopping_rounds': 50,
        'verbosity': 1
    }

    logger.info(f"  scale_pos_weight: {scale_pos_weight:.3f}")
    logger.info(f"  Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    model = xgb.XGBClassifier(**params)

    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=True
    )

    logger.info("[OK] Training complete")

    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""
    logger.info("Evaluating model...")

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Classification report
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_test, y_pred, target_names=['SL', 'TP']))

    # AUC score
    auc = roc_auc_score(y_test, y_pred_proba)
    logger.info(f"\nAUC Score: {auc:.4f}")

    # Accuracy
    accuracy = (y_pred == y_test).mean()
    logger.info(f"Accuracy: {accuracy*100:.2f}%")

    # TP prediction rate (probability > 0.37 baseline)
    tp_predictions = (y_pred_proba > 0.37).sum()
    logger.info(f"\nTP Predictions (prob > 0.37): {tp_predictions} / {len(y_test)} ({tp_predictions/len(y_test)*100:.1f}%)")

    return {
        'auc': auc,
        'accuracy': accuracy,
        'tp_predictions': tp_predictions
    }


def save_model(model, feature_cols):
    """Save trained model and feature columns."""
    logger.info("Saving model...")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = MODEL_DIR / MODEL_NAME
    joblib.dump(model, model_path)
    logger.info(f"  [OK] Model saved: {model_path}")

    # Save feature columns
    feature_path = MODEL_DIR / MODEL_NAME.replace('.joblib', '_features.pkl')
    joblib.dump(feature_cols, feature_path)
    logger.info(f"  [OK] Features saved: {feature_path}")

    # Save feature importance
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    importance_path = MODEL_DIR / MODEL_NAME.replace('.joblib', '_importance.csv')
    importance_df.to_csv(importance_path, index=False)
    logger.info(f"  [OK] Feature importance saved: {importance_path}")

    # Display top 20 features
    logger.info("\nTop 20 Most Important Features:")
    for i, row in importance_df.head(20).iterrows():
        logger.info(f"  {i+1:2d}. {row['feature']:40s} {row['importance']:.4f}")


def main():
    """Main training pipeline."""
    logger.info("="*80)
    logger.info("BTC XGBOOST V12 - ENHANCED MULTI-TIMEFRAME TRAINING")
    logger.info("="*80)

    try:
        # Load data
        logger.info(f"\n[1/6] Loading data from {DATA_FILE}...")
        df = pd.read_csv(DATA_FILE)
        df['date'] = pd.to_datetime(df['date'])
        logger.info(f"  [OK] Loaded {len(df)} rows")
        logger.info(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        logger.info(f"  Existing features: {len(df.columns)}")

        # Create advanced features
        logger.info(f"\n[2/6] Creating advanced multi-timeframe features...")
        df = create_advanced_multi_tf_features(df)
        logger.info(f"  [OK] Total features after enhancement: {len(df.columns)}")

        # Create labels
        logger.info(f"\n[3/6] Creating TP/SL labels...")
        df = create_tp_sl_labels(df)

        # Prepare features and labels
        logger.info(f"\n[4/6] Preparing training data...")
        exclude_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'label',
                       '1d_close', '4h_close', '1w_close', '1d_open', '4h_open', '1w_open',
                       '1d_high', '4h_high', '1w_high', '1d_low', '4h_low', '1w_low',
                       '1d_volume', '4h_volume', '1w_volume']

        # Get candidate feature columns
        candidate_cols = [col for col in df.columns if col not in exclude_cols]

        # Filter to keep only numeric columns
        feature_cols = []
        for col in candidate_cols:
            if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                feature_cols.append(col)
            else:
                logger.info(f"  [SKIP] Non-numeric column: {col} (dtype: {df[col].dtype})")

        logger.info(f"  [OK] Final feature count: {len(feature_cols)} (numeric only)")

        X = df[feature_cols].values
        y = df['label'].values

        # Split data (time-series split)
        split_idx = int(len(X) * (1 - TEST_SIZE))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        logger.info(f"  Train: {len(X_train)} samples")
        logger.info(f"  Test:  {len(X_test)} samples")

        # Train model
        logger.info(f"\n[5/6] Training XGBoost model...")
        model = train_xgboost(X_train, y_train, X_test, y_test)

        # Evaluate
        logger.info(f"\n[6/6] Evaluating model...")
        metrics = evaluate_model(model, X_test, y_test)

        # Save model
        logger.info(f"\n[SAVE] Saving model...")
        save_model(model, feature_cols)

        logger.info("\n" + "="*80)
        logger.info("TRAINING COMPLETE - BTC V12 ENHANCED")
        logger.info("="*80)
        logger.info(f"AUC: {metrics['auc']:.4f}")
        logger.info(f"Accuracy: {metrics['accuracy']*100:.2f}%")
        logger.info(f"Total Features: {len(feature_cols)}")
        logger.info("\nNext step: Run backtest with backtest_btc_xgboost_v12.py")
        logger.info("="*80)

    except Exception as e:
        logger.error(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
