"""
BTC XGBoost V12 - REALTIME BACKTEST WITH ENHANCED MULTI-TIMEFRAME FEATURES
============================================================================

TEST 2: Multi-Timeframe Enhanced Features

This backtest simulates realtime trading with the V12 model that includes
30 new advanced multi-timeframe features.

Baseline (V11): +22.56% ROI (104 trades, 42.3% win rate)
Target: Beat baseline with better feature engineering

Date: 2026-03-29
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
CRYPTO = 'BTC'
BASE_DIR = Path(__file__).parent

# Backtest parameters
BACKTEST_START = '2025-01-01'
BACKTEST_END = '2025-12-31'
INITIAL_CAPITAL = 1000
POSITION_SIZE = 0.95
TP_PCT = 0.015  # 1.5%
SL_PCT = 0.0075  # 0.75%
TRADING_FEE = 0.001
SLIPPAGE = 0.0005

# Model threshold
PREDICTION_THRESHOLD = 0.5


def create_advanced_multi_tf_features(df):
    """
    Create the same advanced multi-timeframe features as used in training.
    MUST match exactly train_btc_xgboost_v12_enhanced.py
    """
    logger.info("Creating advanced multi-timeframe features...")

    # 1. TIMEFRAME ALIGNMENT INDICATORS
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
    ) / 3.0

    df['momentum_all_positive'] = ((df['1d_momentum_5'] > 0) &
                                    (df['4h_momentum_5'] > 0) &
                                    (df['1w_momentum_5'] > 0)).astype(int)

    df['momentum_all_negative'] = ((df['1d_momentum_5'] < 0) &
                                    (df['4h_momentum_5'] < 0) &
                                    (df['1w_momentum_5'] < 0)).astype(int)

    df['trend_all_bullish'] = ((df['close'] > df['1d_ema_50']) &
                                (df['close'] > df['4h_ema_50']) &
                                (df['close'] > df['1w_ema_50'])).astype(int)

    df['trend_all_bearish'] = ((df['close'] < df['1d_ema_50']) &
                                (df['close'] < df['4h_ema_50']) &
                                (df['close'] < df['1w_ema_50'])).astype(int)

    df['momentum_alignment_score'] = (
        (df['1d_momentum_5'] > 0).astype(int) +
        (df['4h_momentum_5'] > 0).astype(int) +
        (df['1w_momentum_5'] > 0).astype(int)
    ) / 3.0

    # 2. CROSS-TIMEFRAME MOMENTUM CORRELATIONS
    df['momentum_1d_4h_corr'] = df['1d_momentum_5'].rolling(20).corr(df['4h_momentum_5'])
    df['momentum_1d_1w_corr'] = df['1d_momentum_5'].rolling(20).corr(df['1w_momentum_5'])
    df['momentum_4h_1w_corr'] = df['4h_momentum_5'].rolling(20).corr(df['1w_momentum_5'])

    df['momentum_divergence_1d_4h'] = np.abs(
        np.sign(df['1d_momentum_5']) - np.sign(df['4h_momentum_5'])
    )
    df['momentum_divergence_1d_1w'] = np.abs(
        np.sign(df['1d_momentum_5']) - np.sign(df['1w_momentum_5'])
    )

    # Momentum strength ratio
    df['momentum_ratio_1d_4h'] = df['1d_momentum_5'] / (df['4h_momentum_5'].abs() + 1e-8)
    df['momentum_ratio_1d_1w'] = df['1d_momentum_5'] / (df['1w_momentum_5'].abs() + 1e-8)

    # 3. VOLUME TREND ALIGNMENT
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

    # 4. VOLATILITY REGIME CLUSTERING
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

    # 5. PRICE ACTION COHERENCE
    # RSI correlation across timeframes
    df['rsi_1d_4h_corr'] = df['1d_rsi_14'].rolling(20).corr(df['4h_rsi_14'])
    df['rsi_1d_1w_corr'] = df['1d_rsi_14'].rolling(20).corr(df['1w_rsi_14'])

    # MACD histogram correlation
    df['macd_1d_4h_corr'] = df['1d_macd_histogram'].rolling(20).corr(df['4h_macd_histogram'])
    df['macd_1d_1w_corr'] = df['1d_macd_histogram'].rolling(20).corr(df['1w_macd_histogram'])

    # 6. MARKET REGIME TRANSITIONS
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

    logger.info(f"[OK] Created advanced multi-timeframe features")

    return df


def load_model():
    """Load trained XGBoost V12 model."""
    logger.info(f"\n{'='*80}")
    logger.info(f"LOADING MODEL")
    logger.info(f"{'='*80}")

    # Model is saved in parent directory's models folder
    model_path = BASE_DIR.parent / 'models' / 'btc_v12_enhanced.joblib'
    feature_path = BASE_DIR.parent / 'models' / 'btc_v12_enhanced_features.pkl'

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature list not found: {feature_path}")

    model = joblib.load(model_path)
    feature_cols = joblib.load(feature_path)

    logger.info(f"[OK] Model loaded: {model_path}")
    logger.info(f"[OK] Features: {len(feature_cols)}")

    return model, feature_cols


def load_and_prepare_data():
    """Load BTC multi-timeframe data and create V12 features."""
    logger.info(f"\n{'='*80}")
    logger.info(f"LOADING DATA")
    logger.info(f"{'='*80}")

    # Data is in parent directory's data folder
    data_file = BASE_DIR.parent / 'data' / 'cache' / 'btc_multi_tf_merged.csv'
    df = pd.read_csv(data_file)
    df['date'] = pd.to_datetime(df['date'])

    logger.info(f"[OK] Loaded {len(df)} rows")
    logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")

    # Create V12 features
    df = create_advanced_multi_tf_features(df)

    logger.info(f"[OK] Total columns after feature creation: {len(df.columns)}")

    return df


def run_backtest(model, feature_cols, df):
    """Run backtest simulating realtime trading."""
    logger.info(f"\n{'='*80}")
    logger.info(f"RUNNING BACKTEST: {BACKTEST_START} to {BACKTEST_END}")
    logger.info(f"{'='*80}")
    logger.info(f"Model: XGBoost V12 Enhanced ({len(feature_cols)} features)")
    logger.info(f"Prediction Threshold: {PREDICTION_THRESHOLD}")

    # Filter backtest period
    df_backtest = df[
        (df['date'] >= BACKTEST_START) &
        (df['date'] <= BACKTEST_END)
    ].copy()

    logger.info(f"Backtest period: {len(df_backtest)} days")

    # Remove NaN
    df_backtest = df_backtest.dropna(subset=feature_cols)
    logger.info(f"After removing NaN: {len(df_backtest)} days")

    # Generate predictions
    X = df_backtest[feature_cols].fillna(0).values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    predictions_proba = model.predict_proba(X)[:, 1]  # Probability of TP (class 1)
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
            # TP signal (prediction == 1)
            if pred == 1:
                # Enter position
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
            # Signal reversal (prediction drops to 0)
            elif pred == 0:
                exit_reason = 'SIGNAL_REVERSAL'
                exit_price = current_price * (1 - SLIPPAGE)

            # Exit position
            if exit_reason:
                exit_value = position['size'] * exit_price * (1 - TRADING_FEE)
                pnl = exit_value - position['position_value']
                pnl_pct = pnl / position['position_value']

                # Update capital
                capital += pnl

                # Record trade
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
    """Display backtest results."""
    logger.info(f"\n{'='*80}")
    logger.info(f"BACKTEST RESULTS - {CRYPTO} XGBoost V12 ENHANCED")
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
    logger.info(f"  V11 BASELINE WAS: +22.56% (104 trades, 42.3% WR)")

    improvement = total_return * 100 - 22.56
    logger.info(f"  IMPROVEMENT:      {improvement:+.2f}%")

    logger.info(f"\nTrade Statistics:")
    logger.info(f"  Total Trades:     {n_trades}")
    logger.info(f"  Wins:             {n_wins} ({win_rate*100:.1f}%)")
    logger.info(f"  Losses:           {n_losses} ({(1-win_rate)*100:.1f}%)")
    logger.info(f"  Avg Win:          {avg_win*100:+.2f}%")
    logger.info(f"  Avg Loss:         {avg_loss*100:.2f}%")

    # Exit reason breakdown
    logger.info(f"\nExit Reasons:")
    for reason, count in trades_df['reason'].value_counts().items():
        logger.info(f"  {reason}: {count} ({count/n_trades*100:.1f}%)")

    logger.info(f"\nTrade Details:")
    for idx, trade in trades_df.iterrows():
        logger.info(f"  Trade {idx+1}: Entry: {trade['entry_date'].date()} @ ${trade['entry_price']:.2f} | " +
                   f"Exit: {trade['exit_date'].date()} @ ${trade['exit_price']:.2f} | " +
                   f"P&L: {trade['pnl_pct']*100:+.2f}% | {trade['reason']}")


def main():
    """Main backtest pipeline."""
    logger.info(f"\n{'='*80}")
    logger.info(f"BTC XGBoost V12 - REALTIME BACKTEST")
    logger.info(f"{'='*80}")

    try:
        # Load model
        model, feature_cols = load_model()

        # Load and prepare data
        df = load_and_prepare_data()

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
