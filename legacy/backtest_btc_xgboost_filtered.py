"""
BTC XGBoost V11 - INTELLIGENT BACKTEST WITH SIGNAL FILTERING
=============================================================

TEST 1: Signal Filtering + Break on Successive Fails

Intelligent filters:
1. Confidence > 0.65 (vs baseline 0.37)
2. Volatility regime (avoid high volatility)
3. Multi-timeframe alignment (1d + 4h + 1w)
4. Volume confirmation
5. Break après 2-3 losses consécutives

Baseline: +22.56% ROI (104 trades, 42.3% win rate)
Target: +30-40% ROI with filtered signals

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

# Backtest parameters (Q1 2026 like ETH/SOL)
BACKTEST_START = '2026-01-01'
BACKTEST_END = '2026-03-24'
INITIAL_CAPITAL = 1000
POSITION_SIZE = 0.95
TP_PCT = 0.015  # 1.5%
SL_PCT = 0.0075  # 0.75%
TRADING_FEE = 0.001
SLIPPAGE = 0.0005

# Signal filtering (INTELLIGENT - BTC adjusted)
MIN_CONFIDENCE = 0.55  # BTC is more volatile, lower threshold
BASELINE_THRESHOLD = 0.37

# Break on fails
MAX_CONSECUTIVE_LOSSES = 2  # Break après 2 pertes
COOLDOWN_DAYS = 7  # Break de 7 jours


def filter_signal_quality(
    prediction_proba,
    features_row,
    min_confidence=0.65
):
    """
    Filter signals based on 5 quality criteria adapted for BTC TP/SL.

    Returns (should_trade, filter_reasons)
    """
    reasons = []

    # 1. CONFIDENCE FILTER
    # BTC uses TP probability, so confidence is the probability itself
    if prediction_proba < min_confidence:
        reasons.append(f"Low confidence: {prediction_proba:.2f}")
        return False, reasons

    # 2. VOLATILITY REGIME (Multi-timeframe) - BTC adjusted
    # BTC is more volatile, relax thresholds
    vol_1d = features_row.get('1d_hist_vol_20', 0)
    vol_4h = features_row.get('4h_hist_vol_20', 0)
    vol_1w = features_row.get('1w_hist_vol_20', 0)

    # High volatility threshold: > 6% daily, > 5% 4h, > 7% weekly (BTC adjusted)
    if vol_1d > 0.06:
        reasons.append(f"High 1d volatility: {vol_1d:.3f}")
        return False, reasons
    if vol_4h > 0.05:
        reasons.append(f"High 4h volatility: {vol_4h:.3f}")
        return False, reasons
    if vol_1w > 0.07:
        reasons.append(f"High 1w volatility: {vol_1w:.3f}")
        return False, reasons

    # 3. MULTI-TIMEFRAME ALIGNMENT (BTC adjusted)
    # At least one timeframe should show positive momentum
    momentum_1d = features_row.get('1d_momentum_5', 0)
    momentum_4h = features_row.get('4h_momentum_5', 0)
    momentum_1w = features_row.get('1w_momentum_5', 0)

    # Require at least 1/3 timeframes positive (BTC is more volatile)
    positive_tf = sum([momentum_1d > 0, momentum_4h > 0, momentum_1w > 0])
    if positive_tf < 1:
        reasons.append(f"Weak TF alignment: {positive_tf}/3 positive")
        return False, reasons

    # 4. VOLUME CONFIRMATION
    # Check volume trend on 1d and 4h
    vol_trend_1d = features_row.get('1d_volume_trend_7', 0)
    vol_trend_4h = features_row.get('4h_volume_trend_7', 0)

    if vol_trend_1d < -0.2 or vol_trend_4h < -0.2:
        reasons.append(f"Declining volume: 1d={vol_trend_1d:.2f}, 4h={vol_trend_4h:.2f}")
        return False, reasons

    # 5. ADX STRENGTH (trend strength)
    adx_1d = features_row.get('1d_adx_14', 0)
    adx_4h = features_row.get('4h_adx_14', 0)

    # Require strong trend (ADX > 25)
    if adx_1d < 25 and adx_4h < 25:
        reasons.append(f"Weak trend: ADX 1d={adx_1d:.1f}, 4h={adx_4h:.1f}")
        return False, reasons

    return True, []


def load_model():
    """Load trained XGBoost model."""
    logger.info(f"\n{'='*80}")
    logger.info(f"LOADING MODEL")
    logger.info(f"{'='*80}")

    model_path = BASE_DIR / 'models' / 'btc_v12_honest_split.joblib'

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = joblib.load(model_path)
    logger.info(f"[OK] Model loaded: {model_path}")

    return model


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

    df['momentum_ratio_1d_4h'] = np.where(
        df['4h_momentum_5'] != 0,
        df['1d_momentum_5'] / df['4h_momentum_5'],
        0
    )
    df['momentum_ratio_1d_1w'] = np.where(
        df['1w_momentum_5'] != 0,
        df['1d_momentum_5'] / df['1w_momentum_5'],
        0
    )

    # 3. VOLATILITY REGIME INDICATORS (6 features)
    df['volatility_all_high'] = (
        (df['1d_hist_vol_20'] > df['1d_hist_vol_20'].rolling(50).quantile(0.75)) &
        (df['4h_hist_vol_20'] > df['4h_hist_vol_20'].rolling(50).quantile(0.75)) &
        (df['1w_hist_vol_20'] > df['1w_hist_vol_20'].rolling(50).quantile(0.75))
    ).astype(int)

    df['volatility_all_low'] = (
        (df['1d_hist_vol_20'] < df['1d_hist_vol_20'].rolling(50).quantile(0.25)) &
        (df['4h_hist_vol_20'] < df['4h_hist_vol_20'].rolling(50).quantile(0.25)) &
        (df['1w_hist_vol_20'] < df['1w_hist_vol_20'].rolling(50).quantile(0.25))
    ).astype(int)

    df['volatility_expanding'] = (
        (df['1d_hist_vol_20'] > df['1d_hist_vol_20'].shift(5)) &
        (df['4h_hist_vol_20'] > df['4h_hist_vol_20'].shift(5))
    ).astype(int)

    df['volatility_contracting'] = (
        (df['1d_hist_vol_20'] < df['1d_hist_vol_20'].shift(5)) &
        (df['4h_hist_vol_20'] < df['4h_hist_vol_20'].shift(5))
    ).astype(int)

    # RSI correlations
    df['rsi_1d_4h_corr'] = df['1d_rsi_14'].rolling(20).corr(df['4h_rsi_14'])
    df['rsi_1d_1w_corr'] = df['1d_rsi_14'].rolling(20).corr(df['1w_rsi_14'])

    # 4. CROSS-TF TREND STRENGTH (4 features)
    df['macd_1d_4h_corr'] = df['1d_macd_line'].rolling(20).corr(df['4h_macd_line'])
    df['macd_1d_1w_corr'] = df['1d_macd_line'].rolling(20).corr(df['1w_macd_line'])

    df['adx_all_weak_trend'] = (
        (df['1d_adx_14'] < 20) &
        (df['4h_adx_14'] < 20) &
        (df['1w_adx_14'] < 20)
    ).astype(int)

    df['adx_strengthening'] = (
        (df['1d_adx_14'] > df['1d_adx_14'].shift(5)) &
        (df['4h_adx_14'] > df['4h_adx_14'].shift(5))
    ).astype(int)

    # 5. VOLUME TREND ALIGNMENT (3 features)
    df['volume_ratio_all_strong'] = (
        (df['1d_volume_ratio_7'] > 1.2) &
        (df['4h_volume_ratio_7'] > 1.2) &
        (df['1w_volume_ratio_7'] > 1.2)
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

    # 6. BOLLINGER BAND WIDTH CORRELATION (1 feature)
    df['bb_width_1d_4h_corr'] = df['1d_bb_width'].rolling(20).corr(df['4h_bb_width'])
    df['bb_width_1d_1w_corr'] = df['1d_bb_width'].rolling(20).corr(df['1w_bb_width'])

    # Fill NaN from rolling correlations
    df = df.fillna(method='ffill').fillna(0)

    logger.info(f"✓ Created 30 advanced features")
    return df


def load_and_prepare_data():
    """Load BTC multi-timeframe data."""
    logger.info(f"\n{'='*80}")
    logger.info(f"LOADING DATA")
    logger.info(f"{'='*80}")

    data_file = BASE_DIR / 'data' / 'cache' / 'btc_multi_tf_merged.csv'
    df = pd.read_csv(data_file)
    df['date'] = pd.to_datetime(df['date'])

    logger.info(f"[OK] Loaded {len(df)} rows")
    logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")

    # Create V12 advanced features
    df = create_advanced_multi_tf_features(df)

    # Display columns
    logger.info(f"Columns: {len(df.columns)}")

    return df


def run_backtest(model, df):
    """Run backtest with intelligent signal filtering + break on fails."""
    logger.info(f"\n{'='*80}")
    logger.info(f"RUNNING BACKTEST: {BACKTEST_START} to {BACKTEST_END}")
    logger.info(f"{'='*80}")
    logger.info(f"Min Confidence: {MIN_CONFIDENCE}")
    logger.info(f"Baseline Threshold: {BASELINE_THRESHOLD}")
    logger.info(f"Max Consecutive Losses: {MAX_CONSECUTIVE_LOSSES}")
    logger.info(f"Cooldown Days: {COOLDOWN_DAYS}")

    # Filter backtest period
    df_backtest = df[
        (df['date'] >= BACKTEST_START) &
        (df['date'] <= BACKTEST_END)
    ].copy()

    logger.info(f"Backtest period: {len(df_backtest)} days")

    # Identify feature columns (exclude non-features)
    exclude_cols = ['date', 'open', 'high', 'low', 'close', 'volume',
                   'label_class', 'label_numeric', 'price_target_pct',
                   'future_price', 'triple_barrier_label']

    feature_cols = [col for col in df_backtest.columns if col not in exclude_cols]

    # Remove NaN
    df_backtest = df_backtest.dropna(subset=feature_cols)
    logger.info(f"After removing NaN: {len(df_backtest)} days")
    logger.info(f"Features: {len(feature_cols)}")

    # Generate predictions
    X = df_backtest[feature_cols].fillna(0).values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    predictions_proba = model.predict_proba(X)[:, 1]  # Probability of TP (class 1)
    predictions = (predictions_proba > BASELINE_THRESHOLD).astype(int)

    df_backtest['prediction'] = predictions
    df_backtest['prediction_proba'] = predictions_proba

    # Initialize backtest
    capital = INITIAL_CAPITAL
    position = None
    trades = []

    total_signals = 0
    filtered_signals = 0
    consecutive_losses = 0
    cooldown_until = None

    for idx, row in df_backtest.iterrows():
        current_date = row['date']
        current_price = row['close']
        pred_proba = row['prediction_proba']

        # Check if in cooldown
        if cooldown_until and current_date < cooldown_until:
            continue  # Skip trading during cooldown
        elif cooldown_until and current_date >= cooldown_until:
            logger.info(f"[COOLDOWN END] {current_date.date()} - Resuming trading")
            cooldown_until = None
            consecutive_losses = 0  # Reset counter

        # No position - check for entry
        if position is None:
            # TP signal (probability > baseline threshold)
            if pred_proba > BASELINE_THRESHOLD:
                total_signals += 1

                # Filter signal quality
                features_row = df_backtest.loc[idx, feature_cols].to_dict()
                should_trade, filter_reasons = filter_signal_quality(
                    pred_proba, features_row, min_confidence=MIN_CONFIDENCE
                )

                if should_trade:
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
                else:
                    filtered_signals += 1

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
            # Signal reversal (probability drops below threshold)
            elif pred_proba < BASELINE_THRESHOLD:
                exit_reason = 'SIGNAL_REVERSAL'
                exit_price = current_price * (1 - SLIPPAGE)

            # Exit position
            if exit_reason:
                exit_value = position['size'] * exit_price * (1 - TRADING_FEE)
                pnl = exit_value - position['position_value']
                pnl_pct = pnl / position['position_value']

                # Update capital
                capital += pnl

                # Track consecutive losses
                if pnl_pct <= 0:
                    consecutive_losses += 1
                else:
                    consecutive_losses = 0  # Reset on win

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

                # Check for cooldown trigger
                if consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
                    cooldown_until = current_date + pd.Timedelta(days=COOLDOWN_DAYS)
                    logger.info(f"[COOLDOWN START] {consecutive_losses} consecutive losses - Break until {cooldown_until.date()}")

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

    return trades, total_signals, filtered_signals, capital


def display_results(trades, total_signals, filtered_signals, final_capital):
    """Display backtest results."""
    logger.info(f"\n{'='*80}")
    logger.info(f"BACKTEST RESULTS - {CRYPTO} XGBoost V11 FILTERED")
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

    logger.info(f"\nSignal Filtering:")
    logger.info(f"  Total signals:    {total_signals}")
    logger.info(f"  Filtered out:     {filtered_signals} ({filtered_signals/total_signals*100:.1f}%)")
    logger.info(f"  Trades executed:  {n_trades} ({n_trades/total_signals*100:.1f}%)")

    logger.info(f"\nOverall Performance:")
    logger.info(f"  Initial Capital:  ${INITIAL_CAPITAL:,.2f}")
    logger.info(f"  Final Capital:    ${final_capital:,.2f}")
    logger.info(f"  Total Return:     {total_return*100:+.2f}%")
    logger.info(f"  BASELINE WAS:     +22.56% (104 trades, 42.3% WR)")

    improvement = total_return * 100 - 22.56
    logger.info(f"  IMPROVEMENT:      {improvement:+.2f}%")

    logger.info(f"\nTrade Statistics:")
    logger.info(f"  Total Trades:     {n_trades}")
    logger.info(f"  Wins:             {n_wins} ({win_rate*100:.1f}%)")
    logger.info(f"  Losses:           {n_losses} ({(1-win_rate)*100:.1f}%)")
    logger.info(f"  Avg Win:          {avg_win*100:+.2f}%")
    logger.info(f"  Avg Loss:         {avg_loss*100:.2f}%")

    # Calculate max consecutive losses
    consecutive_losses = 0
    max_consecutive_losses = 0
    for pnl_pct in trades_df['pnl_pct']:
        if pnl_pct <= 0:
            consecutive_losses += 1
            max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
        else:
            consecutive_losses = 0

    logger.info(f"  Max Consecutive Losses: {max_consecutive_losses}")

    logger.info(f"\nTrade Details:")
    for idx, trade in trades_df.iterrows():
        logger.info(f"  Trade {idx+1}: Entry: {trade['entry_date'].date()} @ ${trade['entry_price']:.2f} | " +
                   f"Exit: {trade['exit_date'].date()} @ ${trade['exit_price']:.2f} | " +
                   f"P&L: {trade['pnl_pct']*100:+.2f}% | {trade['reason']}")


def main():
    """Main backtest pipeline."""
    logger.info(f"\n{'='*80}")
    logger.info(f"BTC XGBoost V11 - TEST 1: INTELLIGENT SIGNAL FILTERING")
    logger.info(f"{'='*80}")

    try:
        # Load model
        model = load_model()

        # Load and prepare data
        df = load_and_prepare_data()

        # Run backtest
        trades, total_signals, filtered_signals, final_capital = run_backtest(model, df)

        # Display results
        display_results(trades, total_signals, filtered_signals, final_capital)

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
