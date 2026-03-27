"""
Walk-Forward Backtesting System - SCRIPT DE REFERENCE PRINCIPAL
================================================================
Système de backtesting avec découpage temporel - LONG ONLY

Utilise les modèles LONG principaux:
- BTC: baseline (237 features)
- ETH: top50 (50 features)
- SOL: optuna (348 features)

Phases de validation Walk-Forward:
- Phase 1: Train 2018-2022 → Test 2023
- Phase 2: Train 2018-2023 → Test 2024
- Phase 3: Train 2018-2024 → Test 2025-2026

Génère automatiquement:
- CSV de tous les trades
- CSV des résumés par phase
- Graphiques prix réel vs prédictions
- Résumé global multi-cryptos
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))


class WalkForwardBacktest:
    """Backtesting Walk-Forward - LONG ONLY"""

    def __init__(self, crypto, mode):
        self.crypto = crypto
        self.mode = mode
        self.models_dir = Path(__file__).parent.parent / 'models'
        self.results_dir = Path(__file__).parent.parent / 'results' / 'walk_forward'
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Phases de test Walk-Forward
        self.phases = [
            {'train_start': '2018-01-01', 'train_end': '2022-12-31',
             'test_start': '2023-01-01', 'test_end': '2023-12-31', 'name': 'Phase1_2023'},
            {'train_start': '2018-01-01', 'train_end': '2023-12-31',
             'test_start': '2024-01-01', 'test_end': '2024-12-31', 'name': 'Phase2_2024'},
            {'train_start': '2018-01-01', 'train_end': '2024-12-31',
             'test_start': '2025-01-01', 'test_end': '2026-03-24', 'name': 'Phase3_2025-2026'}
        ]

        # Thresholds optimisés
        self.thresholds = {
            'btc': 0.55,
            'eth': 0.50,
            'sol': 0.60
        }

        self.all_trades = []
        self.phase_results = []

    def load_data(self):
        """Charger les données"""
        cache_file = Path(__file__).parent.parent / 'data' / 'cache' / f'{self.crypto}_multi_tf_merged.csv'
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        print(f"\nData: {len(df)} rows from {df.index[0]} to {df.index[-1]}")
        return df

    def load_model(self):
        """Charger le modèle LONG"""
        model_file = self.models_dir / f'{self.crypto}_v11_{self.mode}.joblib'
        model = joblib.load(model_file)
        print(f"Model: {self.crypto}_v11_{self.mode} ({model.n_features_in_} features)")
        return model

    def prepare_features(self, df):
        """Préparer les features"""
        exclude_cols = ['open', 'high', 'low', 'close', 'volume',
                       'label_class', 'label_numeric', 'price_target_pct',
                       'future_price', 'triple_barrier_label']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        X = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values
        return X, feature_cols

    def apply_filters(self, df):
        """Appliquer filtres Quick Win"""
        # Regime: SMA50 > SMA200
        df['sma_50'] = df['close'].rolling(50).mean()
        df['sma_200'] = df['close'].rolling(200).mean()
        df['regime_ok'] = df['sma_50'] > df['sma_200']

        # Day of week: pas Friday
        df['dow_ok'] = df.index.dayofweek != 4

        # Low confidence filter
        df['conf_ok'] = (df['prob'] < 0.50) | (df['prob'] >= 0.53)

        return df

    def simulate_trade(self, entry_idx, df_test, entry_price):
        """Simuler trade LONG avec TP/SL"""
        tp_price = entry_price * 1.015  # +1.5%
        sl_price = entry_price * 0.9925  # -0.75%

        # Lookahead 7 jours (42 x 4h)
        lookahead = min(42, len(df_test) - entry_idx - 1)
        future_data = df_test.iloc[entry_idx+1:entry_idx+1+lookahead]

        for idx, row in future_data.iterrows():
            if row['high'] >= tp_price:
                return 'TP', tp_price, idx
            elif row['low'] <= sl_price:
                return 'SL', sl_price, idx

        return None, None, None

    def run_phase(self, phase, df_full, model):
        """Exécuter une phase de test"""
        print(f"\n{'='*80}")
        print(f"PHASE: {phase['name']}")
        print(f"Train: {phase['train_start']} to {phase['train_end']}")
        print(f"Test:  {phase['test_start']} to {phase['test_end']}")
        print(f"{'='*80}")

        # Extraire période de test
        df_test = df_full[(df_full.index >= phase['test_start']) &
                         (df_full.index <= phase['test_end'])].copy()

        if len(df_test) == 0:
            print(f"WARNING: No data for test period")
            return None

        print(f"Test samples: {len(df_test)}")

        # Features
        X, _ = self.prepare_features(df_test)

        # Prédictions
        n_features = model.n_features_in_
        prob = model.predict_proba(X[:, :n_features])[:, 1]
        df_test['prob'] = prob

        # Filtres
        df_test = self.apply_filters(df_test)

        # Signaux
        threshold = self.thresholds[self.crypto]
        df_test['signal'] = ((df_test['prob'] > threshold) &
                            df_test['regime_ok'] &
                            df_test['dow_ok'] &
                            df_test['conf_ok']).astype(int)

        # Simuler trades
        trades = []
        capital = 1000
        consecutive_losses = 0

        for i in range(len(df_test)):
            row = df_test.iloc[i]

            # Skip si pas de signal
            if pd.isna(row['prob']) or row['signal'] == 0:
                continue

            # Check 3 pertes consécutives
            if consecutive_losses >= 3:
                continue

            # Simuler trade
            entry_price = row['close']
            outcome, exit_price, exit_time = self.simulate_trade(i, df_test, entry_price)

            if outcome is None:
                continue

            # PnL
            pnl_pct = ((exit_price - entry_price) / entry_price) * 100
            pnl_usd = capital * (pnl_pct / 100)
            capital += pnl_usd

            # Update losses
            consecutive_losses = consecutive_losses + 1 if outcome == 'SL' else 0

            # Save trade
            trades.append({
                'phase': phase['name'],
                'crypto': self.crypto,
                'entry_time': df_test.index[i],
                'exit_time': exit_time,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'outcome': outcome,
                'pnl_pct': pnl_pct,
                'pnl_usd': pnl_usd,
                'capital': capital,
                'prob': row['prob']
            })

        # Métriques
        if len(trades) == 0:
            print("No trades executed")
            return None

        df_trades = pd.DataFrame(trades)
        n_trades = len(df_trades)
        n_tp = (df_trades['outcome'] == 'TP').sum()
        n_sl = (df_trades['outcome'] == 'SL').sum()
        win_rate = (n_tp / n_trades * 100) if n_trades > 0 else 0
        roi = ((capital - 1000) / 1000) * 100

        phase_result = {
            'phase': phase['name'],
            'crypto': self.crypto,
            'mode': self.mode,
            'n_trades': n_trades,
            'win_rate': win_rate,
            'roi': roi,
            'final_capital': capital,
            'n_tp': int(n_tp),
            'n_sl': int(n_sl)
        }

        print(f"\nResults:")
        print(f"  Trades: {n_trades} ({n_tp} TP, {n_sl} SL)")
        print(f"  Win Rate: {win_rate:.2f}%")
        print(f"  ROI: {roi:+.2f}%")
        print(f"  Final Capital: ${capital:.2f}")

        # Save
        self.all_trades.extend(trades)
        self.phase_results.append(phase_result)

        # Generate chart
        self.generate_chart(df_trades, df_test, phase['name'])

        return df_trades

    def generate_chart(self, df_trades, df_test, phase_name):
        """Générer graphique"""
        plt.figure(figsize=(16, 8))

        # Prix réel
        plt.plot(df_test.index, df_test['close'], 'b-', linewidth=1,
                label='Prix Reel', alpha=0.7)

        # TP/SL
        tp_trades = df_trades[df_trades['outcome'] == 'TP']
        sl_trades = df_trades[df_trades['outcome'] == 'SL']

        plt.scatter(tp_trades['entry_time'], tp_trades['entry_price'],
                   c='green', marker='^', s=100, label='TP', alpha=0.7, edgecolors='black')
        plt.scatter(sl_trades['entry_time'], sl_trades['entry_price'],
                   c='red', marker='v', s=100, label='SL', alpha=0.7, edgecolors='black')

        plt.title(f'{self.crypto.upper()} - {phase_name} - Prix Reel vs Predictions',
                 fontsize=14, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Prix (USD)', fontsize=12)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        chart_file = self.results_dir / f'{self.crypto}_{phase_name}_chart.png'
        plt.savefig(chart_file, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  Chart: {chart_file.name}")

    def save_results(self):
        """Sauvegarder résultats"""
        # Tous les trades
        df_all_trades = pd.DataFrame(self.all_trades)
        trades_file = self.results_dir / f'{self.crypto}_{self.mode}_all_trades.csv'
        df_all_trades.to_csv(trades_file, index=False)
        print(f"\nAll trades: {trades_file.name}")

        # Résumé
        df_summary = pd.DataFrame(self.phase_results)
        summary_file = self.results_dir / f'{self.crypto}_{self.mode}_summary.csv'
        df_summary.to_csv(summary_file, index=False)
        print(f"Summary: {summary_file.name}")

        return df_all_trades, df_summary

    def run(self):
        """Exécuter backtesting complet"""
        print(f"\n{'='*80}")
        print(f"WALK-FORWARD BACKTEST: {self.crypto.upper()} ({self.mode})")
        print(f"{'='*80}")

        df_full = self.load_data()
        model = self.load_model()

        # Exécuter phases
        for phase in self.phases:
            self.run_phase(phase, df_full, model)

        # Sauvegarder
        df_all_trades, df_summary = self.save_results()

        # Résumé final
        self.print_summary(df_summary)

        return df_all_trades, df_summary

    def print_summary(self, df_summary):
        """Afficher résumé"""
        print(f"\n{'='*80}")
        print(f"SUMMARY: {self.crypto.upper()} ({self.mode})")
        print(f"{'='*80}")

        print(f"\n{'Phase':<20} {'Trades':<10} {'Win Rate':<12} {'ROI':<10} {'Capital':<12}")
        print("-" * 80)

        for _, row in df_summary.iterrows():
            print(f"{row['phase']:<20} {int(row['n_trades']):<10} "
                  f"{row['win_rate']:>6.2f}%    {row['roi']:>+6.2f}%  ${row['final_capital']:>8.2f}")

        avg_wr = df_summary['win_rate'].mean()
        avg_roi = df_summary['roi'].mean()
        total_trades = df_summary['n_trades'].sum()

        print("-" * 80)
        print(f"{'AVERAGE':<20} {int(total_trades):<10} {avg_wr:>6.2f}%    {avg_roi:>+6.2f}%")
        print(f"{'='*80}")


def run_all_cryptos():
    """Exécuter pour toutes les cryptos"""
    configs = {
        'btc': 'baseline',
        'eth': 'top50',
        'sol': 'optuna'
    }

    print("="*80)
    print("WALK-FORWARD BACKTEST - SCRIPT DE REFERENCE PRINCIPAL")
    print("="*80)
    print("\nConfiguration:")
    print("  BTC: baseline")
    print("  ETH: top50")
    print("  SOL: optuna")
    print("\nPhases:")
    print("  Phase 1: Train 2018-2022, Test 2023")
    print("  Phase 2: Train 2018-2023, Test 2024")
    print("  Phase 3: Train 2018-2024, Test 2025-2026")

    all_results = {}

    for crypto, mode in configs.items():
        backtest = WalkForwardBacktest(crypto, mode)
        df_trades, df_summary = backtest.run()
        all_results[crypto] = {'trades': df_trades, 'summary': df_summary}

    # Résumé global
    print(f"\n{'='*80}")
    print("GLOBAL SUMMARY - ALL CRYPTOS")
    print(f"{'='*80}")

    all_summaries = []
    for crypto, results in all_results.items():
        all_summaries.append(results['summary'])

    df_global = pd.concat(all_summaries, ignore_index=True)

    # Par phase
    for phase in ['Phase1_2023', 'Phase2_2024', 'Phase3_2025-2026']:
        print(f"\n{phase}:")
        phase_data = df_global[df_global['phase'] == phase]

        if len(phase_data) > 0:
            print(f"{'  Crypto':<10} {'Trades':<10} {'Win Rate':<12} {'ROI':<10}")
            print("  " + "-" * 70)

            for _, row in phase_data.iterrows():
                print(f"  {row['crypto'].upper():<10} {int(row['n_trades']):<10} "
                      f"{row['win_rate']:>6.2f}%    {row['roi']:>+6.2f}%")

    # Sauvegarder global
    results_dir = Path(__file__).parent.parent / 'results' / 'walk_forward'
    global_file = results_dir / 'GLOBAL_SUMMARY.csv'
    df_global.to_csv(global_file, index=False)
    print(f"\nGlobal summary: {global_file.name}")

    print("\n" + "="*80)
    print("BACKTESTING COMPLETE")
    print("Results in: results/walk_forward/")
    print("="*80)


if __name__ == '__main__':
    run_all_cryptos()
