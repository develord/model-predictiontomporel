"""
ETH MODEL - INTELLIGENT BACKTEST WITH SIGNAL FILTERING
======================================================

Backtest with 5-criteria intelligent signal filtering:
1. Model confidence threshold (min_confidence=0.65)
2. Volume confirmation (>70% of average)
3. Momentum alignment (bullish/bearish confirmation)
4. Volatility regime filtering (avoid high volatility)
5. Market structure alignment

Date: 2026-03-29
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import logging

# Import local feature engineering modules
from enhanced_features_enriched import create_enhanced_features
from advanced_features_nontechnical import create_advanced_nontechnical_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
CRYPTO = 'ETH'
BASE_DIR = Path(__file__).parent

# Backtest parameters
BACKTEST_START = '2026-01-01'
BACKTEST_END = '2026-03-31'
INITIAL_CAPITAL = 10000
POSITION_SIZE = 0.95
STOP_LOSS = 0.05
TAKE_PROFIT = 0.10
TRADING_FEE = 0.001
SLIPPAGE = 0.0005

# Signal filtering
MIN_CONFIDENCE = 0.65


def filter_signal_quality(
    prediction_proba,
    signal_type,
    features_row,
    min_confidence=0.65,
    verbose=False
):
    """
    Filter signals based on 5 quality criteria.
    Returns (should_trade, filter_reasons)
    """
    reasons = []

    # 1. CONFIDENCE FILTER
    confidence = abs(prediction_proba - 0.5) * 2
    if confidence < (min_confidence - 0.5) * 2:
        reasons.append(f"Low confidence: {confidence:.2f}")
        return False, reasons

    # 2. VOLUME FILTER
    vol_relative = features_row.get('volume_relative', 1.0)
    if vol_relative < 0.7:
        reasons.append(f"Low volume: {vol_relative:.2f}")
        return False, reasons

    # 3. MOMENTUM ALIGNMENT
    if signal_type == 'LONG':
        momentum_shift_bullish = features_row.get('momentum_shift_bullish', 0)
        if momentum_shift_bullish == 0:
            reasons.append("No bullish momentum shift")
            return False, reasons
    else:  # SHORT
        momentum_shift_bearish = features_row.get('momentum_shift_bearish', 0)
        if momentum_shift_bearish == 0:
            reasons.append("No bearish momentum shift")
            return False, reasons

    # 4. VOLATILITY REGIME
    vol_regime_high = features_row.get('vol_regime_high', 0)
    if vol_regime_high == 1:
        reasons.append("High volatility regime")
        return False, reasons

    # 5. MARKET STRUCTURE
    market_structure_score = features_row.get('market_structure_score', 0)
    if signal_type == 'LONG' and market_structure_score < -0.4:
        reasons.append(f"Bearish market structure: {market_structure_score:.2f}")
        return False, reasons
    elif signal_type == 'SHORT' and market_structure_score > 0.4:
        reasons.append(f"Bullish market structure: {market_structure_score:.2f}")
        return False, reasons

    return True, []


def load_model():
    """
    Load trained model and feature columns.
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"LOADING MODEL")
    logger.info(f"{'='*80}")

    model_dir = BASE_DIR / 'models' / 'xgboost_ultimate'

    # Load model
    model_path = model_dir / 'model.pkl'
    model = joblib.load(model_path)
    logger.info(f"[OK] Model loaded: {model_path}")

    # Load feature columns
    feature_path = model_dir / 'feature_columns.pkl'
    feature_cols = joblib.load(feature_path)
    logger.info(f"[OK] Feature columns loaded: {len(feature_cols)} features")

    return model, feature_cols


def load_and_prepare_data():
    """
    Load data and create features.
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"LOADING AND PREPARING DATA")
    logger.info(f"{'='*80}")

    # Load data
    data_file = BASE_DIR / 'data' / f'{CRYPTO}_1d.csv'
    df = pd.read_csv(data_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    logger.info(f"[OK] Loaded {len(df)} rows")

    # Create features
    logger.info("Creating features...")
    df = create_enhanced_features(df)
    df = create_advanced_nontechnical_features(df)

    logger.info(f"[OK] Total columns: {len(df.columns)}")

    return df


def run_backtest(model, feature_cols, df):
    """
    Run backtest with intelligent signal filtering.
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"RUNNING BACKTEST: {BACKTEST_START} to {BACKTEST_END}")
    logger.info(f"{'='*80}")

    # Filter backtest period
    df_backtest = df[
        (df['timestamp'] >= BACKTEST_START) &
        (df['timestamp'] <= BACKTEST_END)
    ].copy()

    logger.info(f"Backtest period: {len(df_backtest)} days")

    # Remove NaN
    df_backtest = df_backtest.dropna(subset=feature_cols)

    # Generate predictions
    X = df_backtest[feature_cols].values
    predictions_proba = model.predict_proba(X)[:, 1]
    predictions = np.where(predictions_proba > 0.5, 1, 0)

    df_backtest['prediction'] = predictions
    df_backtest['prediction_proba'] = predictions_proba

    # Initialize backtest
    capital = INITIAL_CAPITAL
    position = None
    trades = []

    total_signals = 0
    filtered_signals = 0

    for idx, row in df_backtest.iterrows():
        current_price = row['close']
        pred_proba = row['prediction_proba']

        # No position - check for entry
        if position is None:
            # LONG signal
            if pred_proba > 0.5 + 0.01:
                signal_type = 'LONG'
                total_signals += 1

                # Filter signal quality
                features_row = row[feature_cols].to_dict()
                should_trade, filter_reasons = filter_signal_quality(
                    pred_proba, signal_type, features_row, min_confidence=MIN_CONFIDENCE
                )

                if should_trade:
                    # Enter LONG position
                    position_value = capital * POSITION_SIZE
                    entry_price = current_price * (1 + SLIPPAGE)
                    size = (position_value / entry_price) * (1 - TRADING_FEE)

                    position = {
                        'type': 'LONG',
                        'entry_date': row['timestamp'],
                        'entry_price': entry_price,
                        'size': size,
                        'stop_loss': entry_price * (1 - STOP_LOSS),
                        'take_profit': entry_price * (1 + TAKE_PROFIT)
                    }

                    logger.info(f"[ENTRY] {signal_type} @ {entry_price:.2f} on {row['timestamp'].date()}")
                else:
                    filtered_signals += 1

            # SHORT signal
            elif pred_proba < 0.5 - 0.01:
                signal_type = 'SHORT'
                total_signals += 1

                # Filter signal quality
                features_row = row[feature_cols].to_dict()
                should_trade, filter_reasons = filter_signal_quality(
                    pred_proba, signal_type, features_row, min_confidence=MIN_CONFIDENCE
                )

                if should_trade:
                    # Enter SHORT position
                    position_value = capital * POSITION_SIZE
                    entry_price = current_price * (1 - SLIPPAGE)
                    size = (position_value / entry_price) * (1 - TRADING_FEE)

                    position = {
                        'type': 'SHORT',
                        'entry_date': row['timestamp'],
                        'entry_price': entry_price,
                        'size': size,
                        'stop_loss': entry_price * (1 + STOP_LOSS),
                        'take_profit': entry_price * (1 - TAKE_PROFIT)
                    }

                    logger.info(f"[ENTRY] {signal_type} @ {entry_price:.2f} on {row['timestamp'].date()}")
                else:
                    filtered_signals += 1

        # Has position - check for exit
        else:
            exit_reason = None
            exit_price = None

            if position['type'] == 'LONG':
                # Stop loss
                if row['low'] <= position['stop_loss']:
                    exit_reason = 'STOP_LOSS'
                    exit_price = position['stop_loss']
                # Take profit
                elif row['high'] >= position['take_profit']:
                    exit_reason = 'TAKE_PROFIT'
                    exit_price = position['take_profit']
                # Signal reversal
                elif pred_proba < 0.5:
                    exit_reason = 'SIGNAL_REVERSAL'
                    exit_price = current_price * (1 - SLIPPAGE)

            else:  # SHORT
                # Stop loss
                if row['high'] >= position['stop_loss']:
                    exit_reason = 'STOP_LOSS'
                    exit_price = position['stop_loss']
                # Take profit
                elif row['low'] <= position['take_profit']:
                    exit_reason = 'TAKE_PROFIT'
                    exit_price = position['take_profit']
                # Signal reversal
                elif pred_proba > 0.5:
                    exit_reason = 'SIGNAL_REVERSAL'
                    exit_price = current_price * (1 + SLIPPAGE)

            # Exit position
            if exit_reason:
                exit_value = position['size'] * exit_price * (1 - TRADING_FEE)

                if position['type'] == 'LONG':
                    pnl = exit_value - (capital * POSITION_SIZE)
                else:  # SHORT
                    pnl = (capital * POSITION_SIZE) - exit_value + (capital * POSITION_SIZE)

                pnl_pct = pnl / (capital * POSITION_SIZE)

                # Update capital
                capital += pnl

                # Record trade
                trades.append({
                    'entry_date': position['entry_date'],
                    'exit_date': row['timestamp'],
                    'position': position['type'],
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'reason': exit_reason
                })

                logger.info(f"[EXIT] {position['type']} @ {exit_price:.2f} on {row['timestamp'].date()} | P&L: {pnl_pct*100:+.2f}% | Reason: {exit_reason}")

                position = None

    # Close any open position at end
    if position:
        row = df_backtest.iloc[-1]
        exit_price = row['close'] * (1 - SLIPPAGE if position['type'] == 'LONG' else 1 + SLIPPAGE)
        exit_value = position['size'] * exit_price * (1 - TRADING_FEE)

        if position['type'] == 'LONG':
            pnl = exit_value - (capital * POSITION_SIZE)
        else:
            pnl = (capital * POSITION_SIZE) - exit_value + (capital * POSITION_SIZE)

        pnl_pct = pnl / (capital * POSITION_SIZE)
        capital += pnl

        trades.append({
            'entry_date': position['entry_date'],
            'exit_date': row['timestamp'],
            'position': position['type'],
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'reason': 'END_OF_PERIOD'
        })

        logger.info(f"[EXIT] {position['type']} @ {exit_price:.2f} (END) | P&L: {pnl_pct*100:+.2f}%")

    return trades, total_signals, filtered_signals, capital


def display_results(trades, total_signals, filtered_signals, final_capital):
    """
    Display backtest results.
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"BACKTEST RESULTS - {CRYPTO}")
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

    logger.info(f"\nTrade Statistics:")
    logger.info(f"  Total Trades:     {n_trades}")
    logger.info(f"  Wins:             {n_wins} ({win_rate*100:.1f}%)")
    logger.info(f"  Losses:           {n_losses} ({(1-win_rate)*100:.1f}%)")
    logger.info(f"  Avg Win:          {avg_win*100:+.2f}%")
    logger.info(f"  Avg Loss:         {avg_loss*100:.2f}%")

    logger.info(f"\nTrade Details:")
    for idx, trade in trades_df.iterrows():
        logger.info(f"  Trade {idx+1}: {trade['position']:5s} | Entry: {trade['entry_date'].date()} @ ${trade['entry_price']:.2f} | Exit: {trade['exit_date'].date()} @ ${trade['exit_price']:.2f} | P&L: {trade['pnl_pct']*100:+.2f}% | {trade['reason']}")


def main():
    """
    Main backtest pipeline.
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"ETH MODEL - INTELLIGENT BACKTEST")
    logger.info(f"{'='*80}")
    logger.info(f"Min Confidence: {MIN_CONFIDENCE}")
    logger.info(f"Position Size:  {POSITION_SIZE*100:.0f}%")
    logger.info(f"Stop Loss:      {STOP_LOSS*100:.0f}%")
    logger.info(f"Take Profit:    {TAKE_PROFIT*100:.0f}%")

    try:
        # Load model
        model, feature_cols = load_model()

        # Load and prepare data
        df = load_and_prepare_data()

        # Run backtest
        trades, total_signals, filtered_signals, final_capital = run_backtest(
            model, feature_cols, df
        )

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
