"""
Simulation Détaillée V11 - Analyse Complète de Trading
========================================================
Simulation avec capital initial de 1000$ par crypto
Calcul détaillé de chaque trade avec gestion du capital
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import json


def simulate_trading_with_capital(crypto: str, model_type='optimized', initial_capital=1000.0,
                                  tp_pct=1.5, sl_pct=0.75, prob_threshold=0.5):
    """
    Simule le trading avec gestion de capital réelle

    Args:
        crypto: 'btc', 'eth', or 'sol'
        model_type: 'baseline' or 'optimized'
        initial_capital: Capital de départ ($)
        tp_pct: Take profit (%)
        sl_pct: Stop loss (%)
        prob_threshold: Seuil P(TP) pour entry

    Returns:
        Résultats détaillés de simulation
    """

    # Load model
    if model_type == 'optimized':
        model_file = Path(__file__).parent.parent / 'models' / f'{crypto}_v11_optimized.joblib'
    else:
        model_file = Path(__file__).parent.parent / 'models' / f'{crypto}_v11_classifier.joblib'

    model = joblib.load(model_file)

    # Load data
    cache_file = Path(__file__).parent.parent / 'data' / 'cache' / f'{crypto}_multi_tf_merged.csv'
    df = pd.read_csv(cache_file, index_col=0, parse_dates=True)

    # Prepare features
    exclude_cols = ['open', 'high', 'low', 'close', 'volume',
                   'label_class', 'label_numeric', 'price_target_pct',
                   'future_price', 'triple_barrier_label']
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    # Test split (last 20%)
    split_idx = int(len(df) * 0.8)
    df_test = df.iloc[split_idx:].copy()

    # Prepare features
    X_test = df_test[feature_cols].fillna(0).values
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    # Predict P(TP)
    prob_tp = model.predict_proba(X_test)[:, 1]
    df_test['prob_tp'] = prob_tp
    df_test['signal'] = (prob_tp > prob_threshold).astype(int)

    # Simulate trades avec capital
    capital = initial_capital
    trades = []
    in_position = False
    entry_idx = None
    entry_price = None
    position_size_usd = None

    max_capital = initial_capital
    min_capital = initial_capital

    for idx in range(len(df_test)):
        row = df_test.iloc[idx]

        # Check entry signal
        if not in_position and row['signal'] == 1 and capital > 0:
            in_position = True
            entry_idx = idx
            entry_price = row['close']
            position_size_usd = capital  # All-in sur chaque trade
            continue

        # Check exit
        if in_position:
            current_price = row['close']
            pnl_pct = ((current_price - entry_price) / entry_price) * 100

            # TP hit
            if pnl_pct >= tp_pct:
                profit_usd = position_size_usd * (tp_pct / 100)
                capital += profit_usd

                trades.append({
                    'entry_idx': entry_idx,
                    'exit_idx': idx,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'pnl_pct': tp_pct,
                    'pnl_usd': profit_usd,
                    'capital_after': capital,
                    'outcome': 'TP',
                    'bars_held': idx - entry_idx
                })

                in_position = False
                max_capital = max(max_capital, capital)
                min_capital = min(min_capital, capital)
                continue

            # SL hit
            if pnl_pct <= -sl_pct:
                loss_usd = position_size_usd * (sl_pct / 100)
                capital -= loss_usd

                trades.append({
                    'entry_idx': entry_idx,
                    'exit_idx': idx,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'pnl_pct': -sl_pct,
                    'pnl_usd': -loss_usd,
                    'capital_after': capital,
                    'outcome': 'SL',
                    'bars_held': idx - entry_idx
                })

                in_position = False
                max_capital = max(max_capital, capital)
                min_capital = min(min_capital, capital)
                continue

    # Close any open position
    if in_position:
        final_price = df_test.iloc[-1]['close']
        final_pnl_pct = ((final_price - entry_price) / entry_price) * 100
        final_pnl_usd = position_size_usd * (final_pnl_pct / 100)
        capital += final_pnl_usd

        trades.append({
            'entry_idx': entry_idx,
            'exit_idx': len(df_test) - 1,
            'entry_price': entry_price,
            'exit_price': final_price,
            'pnl_pct': final_pnl_pct,
            'pnl_usd': final_pnl_usd,
            'capital_after': capital,
            'outcome': 'OPEN',
            'bars_held': len(df_test) - 1 - entry_idx
        })

        max_capital = max(max_capital, capital)
        min_capital = min(min_capital, capital)

    # Analyze trades
    if len(trades) == 0:
        return {
            'crypto': crypto,
            'model_type': model_type,
            'initial_capital': initial_capital,
            'final_capital': capital,
            'total_trades': 0,
            'message': 'No trades executed'
        }

    trades_df = pd.DataFrame(trades)

    tp_trades = trades_df[trades_df['outcome'] == 'TP']
    sl_trades = trades_df[trades_df['outcome'] == 'SL']

    # Find best and worst trades
    best_trade = trades_df.loc[trades_df['pnl_usd'].idxmax()]
    worst_trade = trades_df.loc[trades_df['pnl_usd'].idxmin()]

    # Calculate metrics
    results = {
        'crypto': crypto.upper(),
        'model_type': model_type,
        'initial_capital': initial_capital,
        'final_capital': capital,
        'net_profit': capital - initial_capital,
        'roi_pct': ((capital - initial_capital) / initial_capital) * 100,

        'total_trades': len(trades_df),
        'tp_trades': len(tp_trades),
        'sl_trades': len(sl_trades),
        'win_rate': len(tp_trades) / len(trades_df) * 100,

        'total_profit_usd': tp_trades['pnl_usd'].sum() if len(tp_trades) > 0 else 0,
        'total_loss_usd': abs(sl_trades['pnl_usd'].sum()) if len(sl_trades) > 0 else 0,
        'avg_win_usd': tp_trades['pnl_usd'].mean() if len(tp_trades) > 0 else 0,
        'avg_loss_usd': abs(sl_trades['pnl_usd'].mean()) if len(sl_trades) > 0 else 0,

        'best_trade_pnl': best_trade['pnl_usd'],
        'best_trade_pct': best_trade['pnl_pct'],
        'worst_trade_pnl': worst_trade['pnl_usd'],
        'worst_trade_pct': worst_trade['pnl_pct'],

        'max_capital': max_capital,
        'min_capital': min_capital,
        'max_drawdown_usd': initial_capital - min_capital,
        'max_drawdown_pct': ((initial_capital - min_capital) / initial_capital) * 100,

        'avg_bars_held': trades_df['bars_held'].mean(),
    }

    return results, trades_df


def create_detailed_table():
    """Crée un tableau détaillé pour tous les cryptos"""

    print("="*120)
    print("SIMULATION DÉTAILLÉE V11 PRO - CAPITAL INITIAL: 1000$")
    print("="*120)

    all_results = []

    for crypto in ['btc', 'eth', 'sol']:
        for model_type in ['baseline', 'optimized']:
            print(f"\nSimulating {crypto.upper()} - {model_type}...")

            try:
                results, trades_df = simulate_trading_with_capital(
                    crypto=crypto,
                    model_type=model_type,
                    initial_capital=1000.0
                )

                all_results.append(results)

            except Exception as e:
                print(f"Error: {e}")
                continue

    # Create summary table
    print("\n\n" + "="*120)
    print("TABLEAU RÉCAPITULATIF COMPLET")
    print("="*120)

    # Header
    print(f"\n{'Crypto':<8} {'Model':<10} {'Capital':<10} {'Capital':<10} {'Profit':<10} {'ROI':<8} "
          f"{'Trades':<8} {'TP':<6} {'SL':<6} {'WR%':<8} "
          f"{'Meilleur':<12} {'Pire':<12} {'Max DD':<10}")
    print(f"{'':8} {'Type':<10} {'Initial':<10} {'Final':<10} {'Net':<10} {'%':<8} "
          f"{'Total':<8} {'Win':<6} {'Loss':<6} {'':<8} "
          f"{'Trade $':<12} {'Trade $':<12} {'$':<10}")
    print("-"*120)

    # Data rows
    for r in all_results:
        if r['total_trades'] == 0:
            continue

        print(f"{r['crypto']:<8} "
              f"{r['model_type']:<10} "
              f"${r['initial_capital']:<9.2f} "
              f"${r['final_capital']:<9.2f} "
              f"${r['net_profit']:<9.2f} "
              f"{r['roi_pct']:<7.1f}% "
              f"{r['total_trades']:<8} "
              f"{r['tp_trades']:<6} "
              f"{r['sl_trades']:<6} "
              f"{r['win_rate']:<7.1f}% "
              f"${r['best_trade_pnl']:<11.2f} "
              f"${r['worst_trade_pnl']:<11.2f} "
              f"${r['max_drawdown_usd']:<9.2f}")

    # Detailed stats per crypto
    print("\n\n" + "="*120)
    print("ANALYSE DÉTAILLÉE PAR CRYPTO")
    print("="*120)

    for crypto in ['btc', 'eth', 'sol']:
        crypto_results = [r for r in all_results if r['crypto'] == crypto.upper()]

        if len(crypto_results) == 0:
            continue

        print(f"\n{crypto.upper()} - ANALYSE COMPARATIVE")
        print("-"*120)

        for r in crypto_results:
            if r['total_trades'] == 0:
                continue

            print(f"\n  {r['model_type'].upper()}:")
            print(f"    Capital: ${r['initial_capital']:.2f} -> ${r['final_capital']:.2f} (ROI: {r['roi_pct']:+.2f}%)")
            print(f"    Trades: {r['total_trades']} total ({r['tp_trades']} TP / {r['sl_trades']} SL) - Win Rate: {r['win_rate']:.1f}%")
            print(f"    P&L: +${r['total_profit_usd']:.2f} (gains) / -${r['total_loss_usd']:.2f} (pertes) = ${r['net_profit']:+.2f} net")
            print(f"    Moyennes: +${r['avg_win_usd']:.2f} par TP / -${r['avg_loss_usd']:.2f} par SL")
            print(f"    Meilleur trade: ${r['best_trade_pnl']:+.2f} ({r['best_trade_pct']:+.2f}%)")
            print(f"    Pire trade: ${r['worst_trade_pnl']:+.2f} ({r['worst_trade_pct']:+.2f}%)")
            print(f"    Capital max atteint: ${r['max_capital']:.2f}")
            print(f"    Capital min atteint: ${r['min_capital']:.2f}")
            print(f"    Max Drawdown: ${r['max_drawdown_usd']:.2f} ({r['max_drawdown_pct']:.1f}%)")
            print(f"    Durée moyenne position: {r['avg_bars_held']:.1f} candles (4h)")

    # Best performers
    print("\n\n" + "="*120)
    print("CLASSEMENT DES MEILLEURES PERFORMANCES")
    print("="*120)

    # Sort by ROI
    sorted_by_roi = sorted([r for r in all_results if r['total_trades'] > 0],
                          key=lambda x: x['roi_pct'], reverse=True)

    print("\n1. Par ROI (%):")
    for i, r in enumerate(sorted_by_roi[:5], 1):
        print(f"   {i}. {r['crypto']} ({r['model_type']}): {r['roi_pct']:+.2f}% "
              f"(${r['initial_capital']:.0f} -> ${r['final_capital']:.2f})")

    # Sort by absolute profit
    sorted_by_profit = sorted([r for r in all_results if r['total_trades'] > 0],
                             key=lambda x: x['net_profit'], reverse=True)

    print("\n2. Par Profit Absolu ($):")
    for i, r in enumerate(sorted_by_profit[:5], 1):
        print(f"   {i}. {r['crypto']} ({r['model_type']}): ${r['net_profit']:+.2f} "
              f"(ROI: {r['roi_pct']:+.1f}%)")

    # Sort by win rate
    sorted_by_wr = sorted([r for r in all_results if r['total_trades'] > 0],
                         key=lambda x: x['win_rate'], reverse=True)

    print("\n3. Par Win Rate (%):")
    for i, r in enumerate(sorted_by_wr[:5], 1):
        print(f"   {i}. {r['crypto']} ({r['model_type']}): {r['win_rate']:.1f}% "
              f"({r['tp_trades']}/{r['total_trades']} trades)")

    # Portfolio simulation (si on investit 1000$ sur chaque crypto optimized)
    print("\n\n" + "="*120)
    print("SIMULATION PORTFOLIO (1000$ sur chaque crypto OPTIMIZED)")
    print("="*120)

    optimized_results = [r for r in all_results if r['model_type'] == 'optimized' and r['total_trades'] > 0]

    total_initial = sum(r['initial_capital'] for r in optimized_results)
    total_final = sum(r['final_capital'] for r in optimized_results)
    total_profit = total_final - total_initial
    portfolio_roi = (total_profit / total_initial) * 100

    print(f"\n  Investment Initial Total: ${total_initial:,.2f}")
    print(f"  Capital Final Total: ${total_final:,.2f}")
    print(f"  Profit Net Total: ${total_profit:+,.2f}")
    print(f"  ROI Portfolio: {portfolio_roi:+.2f}%")

    print(f"\n  Détail par crypto:")
    for r in optimized_results:
        contribution = r['net_profit']
        print(f"    {r['crypto']}: ${r['initial_capital']:.0f} -> ${r['final_capital']:.2f} "
              f"({r['roi_pct']:+.1f}%) - Contribution: ${contribution:+.2f}")

    # Save results to JSON
    results_dir = Path(__file__).parent.parent / 'backtesting' / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)

    output_file = results_dir / 'detailed_simulation_1000usd.json'

    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n\n  Résultats sauvegardés: {output_file}")

    print("\n" + "="*120)
    print("SIMULATION TERMINÉE")
    print("="*120)

    return all_results


if __name__ == '__main__':
    results = create_detailed_table()
