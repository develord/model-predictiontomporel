"""
BTC XGBoost V12 - PRODUCTION BACKTEST (SAME AS ETH/SOL)
========================================================

Strategy: Same approach as successful ETH/SOL models
- Train: 2018-01-06 → 2025-12-31
- Backtest: 2026-01-01 → 2026-03-24 (Q1 2026)

Goal: Achieve similar performance to ETH/SOL
- ETH: +6.24% (1 trade, 100% win rate)
- SOL: +6.24% (1 trade, 100% win rate)

Date: 2026-03-29
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from pathlib import Path
import logging
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
CRYPTO = 'BTC'
BASE_DIR = Path(__file__).parent

# Time splits (SAME AS ETH/SOL)
TRAIN_START = '2018-01-06'
TRAIN_END = '2025-12-31'
BACKTEST_START = '2026-01-01'
BACKTEST_END = '2026-03-24'

# Trading parameters
INITIAL_CAPITAL = 1000
POSITION_SIZE = 0.95
TP_PCT = 0.015  # 1.5%
SL_PCT = 0.0075  # 0.75%
TRADING_FEE = 0.001
SLIPPAGE = 0.0005
PREDICTION_THRESHOLD = 0.5

# XGBoost Parameters (same as V12)
PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': ['auc', 'logloss'],
    'max_depth': 7,
    'learning_rate': 0.03,
    'n_estimators': 500,
    'min_child_weight': 5,
    'gamma': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': 0.786,
    'random_state': 42,
    'n_jobs': -1
}

EARLY_STOPPING_ROUNDS = 50


def create_advanced_multi_tf_features(df):
    """
    Create the same advanced multi-timeframe features as V12.
    30 new features beyond existing 284 features.
    """
    logger.info("Creating 30 advanced multi-timeframe features...")

    # 1. TIMEFRAME ALIGNMENT INDICATORS (8 features)
    df['trend_all_bullish'] = (
        (df['1d_trend_5'] > 0) &
        (df['4h_trend_5'] > 0) &
        (df['1w_trend_5'] > 0)
    ).astype(int)

    df['trend_all_bearish'] = (
        (df['1d_trend_5'] < 0) &
        (df['4h_trend_5'] < 0) &
        (df['1w_trend_5'] < 0)
    ).astype(int)

    df['momentum_all_positive'] = (
        (df['1d_momentum_5'] > 0) &
        (df['4h_momentum_5'] > 0) &
        (df['1w_momentum_5'] > 0)
    ).astype(int)

    df['momentum_all_negative'] = (
        (df['1d_momentum_5'] < 0) &
        (df['4h_momentum_5'] < 0) &
        (df['1w_momentum_5'] < 0)
    ).astype(int)

    df['rsi_all_overbought'] = (
        (df['1d_rsi_14'] > 70) &
        (df['4h_rsi_14'] > 70) &
        (df['1w_rsi_14'] > 70)
    ).astype(int)

    df['rsi_all_oversold'] = (
        (df['1d_rsi_14'] < 30) &
        (df['4h_rsi_14'] < 30) &
        (df['1w_rsi_14'] < 30)
    ).astype(int)

    df['rsi_alignment_score'] = (
        (df['1d_rsi_14'] > 50).astype(int) +
        (df['4h_rsi_14'] > 50).astype(int) +
        (df['1w_rsi_14'] > 50).astype(int)
    ) / 3.0

    df['adx_all_strong_trend'] = (
        (df['1d_adx_14'] > 25) &
        (df['4h_adx_14'] > 25) &
        (df['1w_adx_14'] > 25)
    ).astype(int)

    # 2. CROSS-TIMEFRAME MOMENTUM CORRELATIONS (7 features)
    df['momentum_1d_4h_corr'] = df['1d_momentum_5'].rolling(20).corr(df['4h_momentum_5'])
    df['momentum_1d_1w_corr'] = df['1d_momentum_5'].rolling(20).corr(df['1w_momentum_5'])
    df['momentum_4h_1w_corr'] = df['4h_momentum_5'].rolling(20).corr(df['1w_momentum_5'])

    df['momentum_divergence_1d_4h'] = df['1d_momentum_5'] - df['4h_momentum_5']
    df['momentum_divergence_1d_1w'] = df['1d_momentum_5'] - df['1w_momentum_5']

    df['momentum_ratio_1d_4h'] = df['1d_momentum_5'] / (df['4h_momentum_5'].abs() + 1e-8)
    df['momentum_ratio_1d_1w'] = df['1d_momentum_5'] / (df['1w_momentum_5'].abs() + 1e-8)

    # 3. VOLUME TREND ALIGNMENT (3 features)
    df['volume_ratio_all_strong'] = (
        (df['1d_volume_ratio_7'] > 1.0) &
        (df['4h_volume_ratio_7'] > 1.0) &
        (df['1w_volume_ratio_7'] > 1.0)
    ).astype(int)

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

    # 4. VOLATILITY REGIME CLUSTERING (4 features)
    df['volatility_all_high'] = (
        (df['1d_hist_vol_20'] > df['1d_hist_vol_20'].quantile(0.75)) &
        (df['4h_hist_vol_20'] > df['4h_hist_vol_20'].quantile(0.75)) &
        (df['1w_hist_vol_20'] > df['1w_hist_vol_20'].quantile(0.75))
    ).astype(int)

    df['volatility_all_low'] = (
        (df['1d_hist_vol_20'] < df['1d_hist_vol_20'].quantile(0.25)) &
        (df['4h_hist_vol_20'] < df['4h_hist_vol_20'].quantile(0.25)) &
        (df['1w_hist_vol_20'] < df['1w_hist_vol_20'].quantile(0.25))
    ).astype(int)

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

    # 5. PRICE ACTION COHERENCE (4 features)
    df['rsi_1d_4h_corr'] = df['1d_rsi_14'].rolling(20).corr(df['4h_rsi_14'])
    df['rsi_1d_1w_corr'] = df['1d_rsi_14'].rolling(20).corr(df['1w_rsi_14'])

    df['macd_1d_4h_corr'] = df['1d_macd_histogram'].rolling(20).corr(df['4h_macd_histogram'])
    df['macd_1d_1w_corr'] = df['1d_macd_histogram'].rolling(20).corr(df['1w_macd_histogram'])

    # 6. MARKET REGIME TRANSITIONS (4 features)
    df['adx_all_weak_trend'] = (
        (df['1d_adx_14'] < 20) &
        (df['4h_adx_14'] < 20) &
        (df['1w_adx_14'] < 20)
    ).astype(int)

    df['adx_strengthening'] = (
        (df['1d_adx_14'] > df['1d_adx_14'].shift(5)) &
        (df['4h_adx_14'] > df['4h_adx_14'].shift(5))
    ).astype(int)

    df['bb_width_1d_4h_corr'] = df['1d_bb_width'].rolling(20).corr(df['4h_bb_width'])
    df['bb_width_1d_1w_corr'] = df['1d_bb_width'].rolling(20).corr(df['1w_bb_width'])

    logger.info("✓ Created 30 advanced features")
    return df


def load_and_prepare_data():
    """Load BTC multi-timeframe data."""
    logger.info(f"\n{'='*80}")
    logger.info(f"LOADING DATA")
    logger.info(f"{'='*80}")

    data_file = BASE_DIR.parent / 'data' / 'cache' / 'btc_multi_tf_merged.csv'
    df = pd.read_csv(data_file)
    df['date'] = pd.to_datetime(df['date'])

    logger.info(f"[OK] Loaded {len(df)} rows")
    logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")

    # Create advanced features
    df = create_advanced_multi_tf_features(df)

    return df


def train_model(df):
    """Train model on 2018-2023 data only."""
    logger.info(f"\n{'='*80}")
    logger.info(f"TRAINING MODEL: {TRAIN_START} to {TRAIN_END}")
    logger.info(f"{'='*80}")

    # Filter training period
    df_train = df[
        (df['date'] >= TRAIN_START) &
        (df['date'] <= TRAIN_END)
    ].copy()

    logger.info(f"Training period: {len(df_train)} days")

    # Identify feature columns
    exclude_cols = ['date', 'open', 'high', 'low', 'close', 'volume',
                   'label_class', 'label_numeric', 'price_target_pct',
                   'future_price', 'triple_barrier_label']

    candidate_cols = [col for col in df_train.columns if col not in exclude_cols]

    # Filter only numeric features
    feature_cols = []
    for col in candidate_cols:
        if df_train[col].dtype in ['int64', 'float64', 'int32', 'float32']:
            feature_cols.append(col)

    logger.info(f"Features: {len(feature_cols)}")

    # Remove NaN ONLY from label (not features - will be filled with 0)
    df_train = df_train.dropna(subset=['triple_barrier_label'])
    logger.info(f"After removing NaN from label: {len(df_train)} days")

    # Check label distribution
    label_counts = df_train['triple_barrier_label'].value_counts()
    logger.info(f"\nLabel distribution:")
    for label, count in label_counts.items():
        logger.info(f"  Label {int(label)}: {count} ({count/len(df_train)*100:.1f}%)")

    # Remap labels: -1 → 0 (for binary classification)
    df_train['triple_barrier_label'] = df_train['triple_barrier_label'].replace(-1, 0)
    logger.info(f"\nRemapped -1 → 0 for binary classification")

    # Prepare data
    X_train = df_train[feature_cols].fillna(0).values
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    y_train = df_train['triple_barrier_label'].values

    # Split for validation (15% of training data)
    split_idx = int(len(X_train) * 0.85)
    X_train_split = X_train[:split_idx]
    y_train_split = y_train[:split_idx]
    X_val_split = X_train[split_idx:]
    y_val_split = y_train[split_idx:]

    logger.info(f"\nSplit breakdown:")
    logger.info(f"  Train: {len(X_train_split)} samples ({TRAIN_START} → ~2022)")
    logger.info(f"    - Class 0 (SL): {(y_train_split == 0).sum()}")
    logger.info(f"    - Class 1 (TP): {(y_train_split == 1).sum()}")
    logger.info(f"  Val:   {len(X_val_split)} samples (~2022 → {TRAIN_END})")
    logger.info(f"    - Class 0 (SL): {(y_val_split == 0).sum()}")
    logger.info(f"    - Class 1 (TP): {(y_val_split == 1).sum()}")

    # Check class distribution in splits
    train_classes = len(np.unique(y_train_split))
    val_classes = len(np.unique(y_val_split))
    logger.info(f"  Train classes: {train_classes}")
    logger.info(f"  Val classes: {val_classes}")

    if train_classes < 2 or val_classes < 2:
        logger.warning("One split has only one class! Using full training set without validation.")
        X_train_split = X_train
        y_train_split = y_train
        X_val_split = None
        y_val_split = None

    # Train XGBoost
    logger.info(f"\nTraining XGBoost...")
    model = xgb.XGBClassifier(**PARAMS)

    if X_val_split is not None:
        model.fit(
            X_train_split, y_train_split,
            eval_set=[(X_val_split, y_val_split)],
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose=False
        )

        # Evaluate on validation
        y_val_pred = model.predict(X_val_split)
        y_val_proba = model.predict_proba(X_val_split)[:, 1]

        acc = accuracy_score(y_val_split, y_val_pred)
        auc = roc_auc_score(y_val_split, y_val_proba)

        logger.info(f"\n[VALIDATION METRICS]")
        logger.info(f"  Accuracy: {acc:.4f}")
        logger.info(f"  AUC: {auc:.4f}")
    else:
        # No validation split - train on all data
        logger.info("Training without validation split (using all data)")
        model.fit(X_train_split, y_train_split, verbose=False)

    # Save model
    model_path = BASE_DIR.parent / 'models' / 'btc_v12_honest_split.joblib'
    feature_path = BASE_DIR.parent / 'models' / 'btc_v12_honest_split_features.pkl'

    joblib.dump(model, model_path)
    joblib.dump(feature_cols, feature_path)

    logger.info(f"\n[OK] Model saved: {model_path}")
    logger.info(f"[OK] Features saved: {feature_path}")

    return model, feature_cols


def run_backtest(model, feature_cols, df):
    """Run backtest on 2024 data (truly unseen)."""
    logger.info(f"\n{'='*80}")
    logger.info(f"RUNNING BACKTEST: {BACKTEST_START} to {BACKTEST_END}")
    logger.info(f"{'='*80}")
    logger.info(f"⚠️  Model has NEVER seen this data!")

    # Filter backtest period
    df_backtest = df[
        (df['date'] >= BACKTEST_START) &
        (df['date'] <= BACKTEST_END)
    ].copy()

    logger.info(f"Backtest period: {len(df_backtest)} days")

    # Generate predictions
    X = df_backtest[feature_cols].fillna(0).values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    predictions_proba = model.predict_proba(X)[:, 1]
    predictions = (predictions_proba > PREDICTION_THRESHOLD).astype(int)

    df_backtest['prediction'] = predictions
    df_backtest['prediction_proba'] = predictions_proba

    # Initialize backtest
    capital = INITIAL_CAPITAL
    position = None
    trades = []

    for idx, row in df_backtest.iterrows():
        current_date = row['date']
        current_price = row['close']
        pred = row['prediction']
        pred_proba = row['prediction_proba']

        # No position - check for entry
        if position is None:
            if pred == 1:  # TP signal
                position_value = capital * POSITION_SIZE
                entry_price = current_price * (1 + SLIPPAGE)
                size = (position_value / entry_price) * (1 - TRADING_FEE)

                tp_price = entry_price * (1 + TP_PCT)
                sl_price = entry_price * (1 - SL_PCT)

                position = {
                    'entry_date': current_date,
                    'entry_price': entry_price,
                    'size': size,
                    'tp_price': tp_price,
                    'sl_price': sl_price,
                    'position_value': position_value
                }

                logger.info(f"[ENTRY] TP signal @ {entry_price:.2f} on {current_date.date()} | Confidence: {pred_proba:.3f}")

        # Has position - check for exit
        else:
            exit_reason = None
            exit_price = None

            # Take profit
            if row['high'] >= position['tp_price']:
                exit_reason = 'TAKE_PROFIT'
                exit_price = position['tp_price']
            # Stop loss
            elif row['low'] <= position['sl_price']:
                exit_reason = 'STOP_LOSS'
                exit_price = position['sl_price']
            # Signal reversal
            elif pred == 0:
                exit_reason = 'SIGNAL_REVERSAL'
                exit_price = current_price * (1 - SLIPPAGE)

            # Exit position
            if exit_reason:
                exit_value = position['size'] * exit_price * (1 - TRADING_FEE)
                pnl = exit_value - position['position_value']
                pnl_pct = pnl / position['position_value']

                capital += pnl

                trades.append({
                    'entry_date': position['entry_date'],
                    'exit_date': current_date,
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'reason': exit_reason
                })

                logger.info(f"[EXIT] @ {exit_price:.2f} on {current_date.date()} | P&L: {pnl_pct*100:+.2f}% | Reason: {exit_reason}")

                position = None

    # Close any open position at end
    if position:
        row = df_backtest.iloc[-1]
        exit_price = row['close'] * (1 - SLIPPAGE)
        exit_value = position['size'] * exit_price * (1 - TRADING_FEE)
        pnl = exit_value - position['position_value']
        pnl_pct = pnl / position['position_value']
        capital += pnl

        trades.append({
            'entry_date': position['entry_date'],
            'exit_date': row['date'],
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'reason': 'END_OF_PERIOD'
        })

        logger.info(f"[EXIT] @ {exit_price:.2f} (END) | P&L: {pnl_pct*100:+.2f}%")

    return trades, capital


def display_results(trades, final_capital):
    """Display backtest results with comparison to biased backtest."""
    logger.info(f"\n{'='*80}")
    logger.info(f"BACKTEST RESULTS - BTC V12 HONEST SPLIT")
    logger.info(f"{'='*80}")

    if not trades:
        logger.info("[WARN] No trades executed")
        return

    trades_df = pd.DataFrame(trades)

    # Overall stats
    total_return = (final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL
    n_trades = len(trades_df)
    n_wins = (trades_df['pnl_pct'] > 0).sum()
    n_losses = (trades_df['pnl_pct'] <= 0).sum()
    win_rate = n_wins / n_trades if n_trades > 0 else 0

    avg_win = trades_df[trades_df['pnl_pct'] > 0]['pnl_pct'].mean() if n_wins > 0 else 0
    avg_loss = trades_df[trades_df['pnl_pct'] <= 0]['pnl_pct'].mean() if n_losses > 0 else 0

    logger.info(f"\nOverall Performance:")
    logger.info(f"  Initial Capital:  ${INITIAL_CAPITAL:,.2f}")
    logger.info(f"  Final Capital:    ${final_capital:,.2f}")
    logger.info(f"  Total Return:     {total_return*100:+.2f}%")

    logger.info(f"\nTrade Statistics:")
    logger.info(f"  Total Trades:     {n_trades}")
    logger.info(f"  Wins:             {n_wins} ({win_rate*100:.1f}%)")
    logger.info(f"  Losses:           {n_losses} ({(1-win_rate)*100:.1f}%)")
    logger.info(f"  Avg Win:          {avg_win*100:+.2f}%")
    logger.info(f"  Avg Loss:         {avg_loss*100:.2f}%")

    logger.info(f"\n{'='*80}")
    logger.info(f"COMPARISON WITH ETH/SOL")
    logger.info(f"{'='*80}")
    logger.info(f"ETH (Q1 2026):")
    logger.info(f"  Return: +6.24%")
    logger.info(f"  Win Rate: 100%")
    logger.info(f"  Trades: 1")
    logger.info(f"\nSOL (Q1 2026):")
    logger.info(f"  Return: +6.24%")
    logger.info(f"  Win Rate: 100%")
    logger.info(f"  Trades: 1")
    logger.info(f"\nBTC (Q1 2026):")
    logger.info(f"  Return: {total_return*100:+.2f}%")
    logger.info(f"  Win Rate: {win_rate*100:.1f}%")
    logger.info(f"  Trades: {n_trades}")

    if n_trades > 0 and total_return > 0:
        logger.info(f"\n✅ BTC model is profitable on Q1 2026!")
    else:
        logger.info(f"\n⚠️  BTC model needs improvement")

    logger.info(f"\nTrade Details:")
    for idx, trade in trades_df.iterrows():
        logger.info(f"  Trade {idx+1}: Entry: {trade['entry_date'].date()} @ ${trade['entry_price']:.2f} | " +
                   f"Exit: {trade['exit_date'].date()} @ ${trade['exit_price']:.2f} | " +
                   f"P&L: {trade['pnl_pct']*100:+.2f}% | {trade['reason']}")


def main():
    """Main pipeline."""
    logger.info(f"\n{'='*80}")
    logger.info(f"BTC V12 - PRODUCTION BACKTEST (SAME AS ETH/SOL)")
    logger.info(f"{'='*80}")
    logger.info(f"Strategy: Train on 2018-2025, Backtest on Q1 2026")

    try:
        # Load data
        df = load_and_prepare_data()

        # Train model
        model, feature_cols = train_model(df)

        # Run backtest
        trades, final_capital = run_backtest(model, feature_cols, df)

        # Display results
        display_results(trades, final_capital)

        logger.info(f"\n{'='*80}")
        logger.info(f"BACKTEST COMPLETE")
        logger.info(f"{'='*80}")

    except Exception as e:
        logger.error(f"\n[ERROR] Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
