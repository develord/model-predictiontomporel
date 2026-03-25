"""
Rapport Détaillé des Trades avec Probabilités
==============================================
Affiche tous les trades avec probabilité du modèle et résultat
"""

import sys
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent))


def generate_detailed_trades(crypto: str, mode: str, tp_pct=1.5, sl_pct=0.75, prob_threshold=0.5):
    """
    Génère un rapport détaillé de tous les trades

    Args:
        crypto: 'btc', 'eth', or 'sol'
        mode: 'baseline', 'optuna', or 'top50'
    """
    print(f"\n{'='*100}")
    print(f"RAPPORT DÉTAILLÉ DES TRADES - {crypto.upper()} {mode.upper()}")
    print('='*100)

    # Load model
    model_file = Path(__file__).parent.parent / 'models' / f'{crypto}_v11_{mode}.joblib'
    stats_file = Path(__file__).parent.parent / 'models' / f'{crypto}_v11_{mode}_stats.json'

    if not model_file.exists():
        print(f"  ERROR: Model not found: {model_file}")
        return None

    model = joblib.load(model_file)

    with open(stats_file, 'r') as f:
        stats = json.load(f)

    # Load data
    cache_file = Path(__file__).parent.parent / 'data' / 'cache' / f'{crypto}_multi_tf_merged.csv'
    df = pd.read_csv(cache_file, index_col=0, parse_dates=True)

    # Prepare features (same as training)
    exclude_cols = [
        'open', 'high', 'low', 'close', 'volume',
        'label_class', 'label_numeric', 'price_target_pct',
        'future_price', 'triple_barrier_label'
    ]

    all_features = [col for col in df.columns if col not in exclude_cols]

    # For TOP50, get exact features
    if mode == 'top50':
        top_features = [f['feature'] for f in stats['top_features'][:50]]
        feature_cols = [f for f in top_features if f in df.columns]
    else:
        feature_cols = all_features

    # Filter to test period (2025+)
    df_test = df[df.index >= '2025-01-01'].copy()

    print(f"  Periode: {df_test.index[0]} to {df_test.index[-1]}")
    print(f"  Features: {len(feature_cols)}")

    # Prepare features
    X_test = df_test[feature_cols].fillna(0).values
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    # Predict probabilities
    prob_tp = model.predict_proba(X_test)[:, 1]
    df_test['prob_tp'] = prob_tp
    df_test['prob_sl'] = 1 - prob_tp
    df_test['signal'] = prob_tp > prob_threshold

    # Simulate trades
    trades = []
    position = None

    for i in range(len(df_test)):
        row = df_test.iloc[i]

        # Skip if in position
        if position is not None:
            # Check if TP/SL hit
            high = df_test['high'].iloc[i]
            low = df_test['low'].iloc[i]

            entry_price = position['entry_price']
            tp_price = entry_price * (1 + tp_pct / 100)
            sl_price = entry_price * (1 - sl_pct / 100)

            hit_tp = high >= tp_price
            hit_sl = low <= sl_price

            if hit_tp or hit_sl:
                # Close position
                exit_price = tp_price if hit_tp else sl_price
                exit_type = 'TP' if hit_tp else 'SL'
                pnl_pct = ((exit_price - entry_price) / entry_price) * 100

                position['exit_time'] = row.name
                position['exit_price'] = exit_price
                position['exit_type'] = exit_type
                position['pnl_pct'] = pnl_pct
                position['bars_held'] = i - position['entry_idx']
                position['result'] = 'WIN' if hit_tp else 'LOSS'

                trades.append(position)
                position = None

        # Open new position if signal and no position
        elif row['signal']:
            position = {
                'trade_id': len(trades) + 1,
                'entry_time': row.name,
                'entry_idx': i,
                'entry_price': row['close'],
                'prob_tp': row['prob_tp'],
                'prob_sl': row['prob_sl'],
                'confidence': abs(row['prob_tp'] - 0.5) * 200  # 0-100% scale
            }

    if len(trades) == 0:
        print("  No trades generated!")
        return None

    # Convert to DataFrame
    trades_df = pd.DataFrame(trades)

    # Format for display
    trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time']).dt.strftime('%Y-%m-%d %H:%M')
    trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time']).dt.strftime('%Y-%m-%d %H:%M')
    trades_df['prob_tp'] = trades_df['prob_tp'].apply(lambda x: f"{x*100:.2f}%")
    trades_df['prob_sl'] = trades_df['prob_sl'].apply(lambda x: f"{x*100:.2f}%")
    trades_df['confidence'] = trades_df['confidence'].apply(lambda x: f"{x:.1f}%")
    trades_df['entry_price'] = trades_df['entry_price'].apply(lambda x: f"${x:,.2f}")
    trades_df['exit_price'] = trades_df['exit_price'].apply(lambda x: f"${x:,.2f}")
    trades_df['pnl_pct'] = trades_df['pnl_pct'].apply(lambda x: f"{x:+.2f}%")

    # Select columns for display
    display_cols = [
        'trade_id', 'entry_time', 'exit_time', 'entry_price', 'exit_price',
        'prob_tp', 'prob_sl', 'confidence', 'exit_type', 'result', 'pnl_pct', 'bars_held'
    ]

    trades_display = trades_df[display_cols].copy()

    # Stats
    total = len(trades_df)
    wins = len(trades_df[trades_df['result'] == 'WIN'])
    losses = len(trades_df[trades_df['result'] == 'LOSS'])
    win_rate = (wins / total) * 100

    # Numeric PnL for stats
    trades_numeric = pd.DataFrame(trades)
    total_pnl = trades_numeric['pnl_pct'].sum()
    avg_win = trades_numeric[trades_numeric['result'] == 'WIN']['pnl_pct'].mean() if wins > 0 else 0
    avg_loss = trades_numeric[trades_numeric['result'] == 'LOSS']['pnl_pct'].mean() if losses > 0 else 0

    # Confidence stats
    avg_confidence = trades_numeric['confidence'].mean()

    print(f"\n{'='*100}")
    print("STATISTIQUES GLOBALES")
    print('='*100)
    print(f"Total Trades: {total}")
    print(f"Wins: {wins} ({win_rate:.2f}%)")
    print(f"Losses: {losses} ({100-win_rate:.2f}%)")
    print(f"ROI Total: {total_pnl:+.2f}%")
    print(f"Avg Win: {avg_win:+.2f}%")
    print(f"Avg Loss: {avg_loss:+.2f}%")
    print(f"Avg Confidence: {avg_confidence:.1f}%")
    print()

    # Save to CSV
    results_dir = Path(__file__).parent.parent / 'backtesting' / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)

    csv_file = results_dir / f'{crypto}_{mode}_detailed_trades.csv'
    trades_df.to_csv(csv_file, index=False)

    print(f"Saved to: {csv_file}")
    print()

    return trades_display, trades_numeric


def generate_all_reports():
    """Generate detailed reports for all selected models"""

    models = [
        ('btc', 'baseline'),
        ('eth', 'top50'),
        ('sol', 'optuna')
    ]

    print("\n" + "="*100)
    print("RAPPORTS DÉTAILLÉS DES TRADES - TOUS LES MODÈLES")
    print("="*100)

    all_trades = {}

    for crypto, mode in models:
        trades_display, trades_numeric = generate_detailed_trades(crypto, mode)

        if trades_display is not None:
            all_trades[f'{crypto}_{mode}'] = {
                'display': trades_display,
                'numeric': trades_numeric
            }

            # Display first 10 and last 10 trades
            print(f"\n{'='*100}")
            print(f"{crypto.upper()} {mode.upper()} - PREMIERS 10 TRADES")
            print('='*100)
            print(trades_display.head(10).to_string(index=False))

            print(f"\n{'='*100}")
            print(f"{crypto.upper()} {mode.upper()} - DERNIERS 10 TRADES")
            print('='*100)
            print(trades_display.tail(10).to_string(index=False))
            print()

    # Combined summary
    print(f"\n{'='*100}")
    print("RÉSUMÉ COMBINÉ - TOUS LES MODÈLES")
    print('='*100)

    summary_rows = []
    for (crypto, mode) in models:
        key = f'{crypto}_{mode}'
        if key in all_trades:
            trades = all_trades[key]['numeric']
            wins = len(trades[trades['result'] == 'WIN'])
            total = len(trades)
            win_rate = (wins / total) * 100
            roi = trades['pnl_pct'].sum()
            avg_conf = trades['confidence'].mean()

            summary_rows.append({
                'Crypto': crypto.upper(),
                'Mode': mode.upper(),
                'Total Trades': total,
                'Wins': wins,
                'Losses': total - wins,
                'Win Rate': f"{win_rate:.2f}%",
                'ROI': f"{roi:+.2f}%",
                'Avg Confidence': f"{avg_conf:.1f}%"
            })

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))

    print(f"\n{'='*100}")
    print("Fichiers CSV sauvegardés dans: backtesting/results/")
    print('='*100)

    return all_trades


if __name__ == '__main__':
    all_trades = generate_all_reports()
