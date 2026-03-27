"""
Comparative Walk-Forward Backtesting: V11 vs V11_PHASES
========================================================
Compare deux approches de training:
- V11: Un seul modèle entraîné sur tout <2025, testé sur chaque phase
- V11_PHASES: Modèle re-entraîné pour chaque phase (walk-forward)

Phases:
- Phase 1: Train 2018-2022 → Test 2023
- Phase 2: Train 2018-2023 → Test 2024
- Phase 3: Train 2018-2024 → Test 2025-2026

LONG-only trading avec Quick Win filters
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))


class ComparativeWalkForward:
    """Compare V11 (single model) vs V11_PHASES (retrained each phase)"""

    def __init__(self, crypto, mode):
        self.crypto = crypto
        self.mode = mode
        self.results_dir = Path(__file__).parent.parent / 'results' / 'compare_v11_phases'
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Walk-forward phases
        self.phases = [
            {'train_start': '2018-01-01', 'train_end': '2022-12-31',
             'test_start': '2023-01-01', 'test_end': '2023-12-31', 'name': 'Phase1_2023'},
            {'train_start': '2018-01-01', 'train_end': '2023-12-31',
             'test_start': '2024-01-01', 'test_end': '2024-12-31', 'name': 'Phase2_2024'},
            {'train_start': '2018-01-01', 'train_end': '2024-12-31',
             'test_start': '2025-01-01', 'test_end': '2026-03-24', 'name': 'Phase3_2025-2026'}
        ]

        # Thresholds
        self.thresholds = {'btc': 0.55, 'eth': 0.50, 'sol': 0.60}

        # XGBoost params (baseline)
        self.xgb_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 200,
            'gamma': 2,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'tree_method': 'hist',
            'verbosity': 0
        }

        self.results = {'v11': [], 'v11_phases': []}

    def load_data(self):
        """Charger les données"""
        cache_file = Path(__file__).parent.parent / 'data' / 'cache' / f'{self.crypto}_multi_tf_merged.csv'
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        print(f"Data: {len(df)} rows from {df.index[0]} to {df.index[-1]}")
        return df

    def prepare_features(self, df):
        """Préparer features et target"""
        exclude_cols = ['open', 'high', 'low', 'close', 'volume',
                       'label_class', 'label_numeric', 'price_target_pct',
                       'future_price', 'triple_barrier_label']

        feature_cols = [col for col in df.columns if col not in exclude_cols]

        # Nettoyer données
        df_clean = df[df['triple_barrier_label'].notna()].copy()

        X = df_clean[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values
        y = (df_clean['triple_barrier_label'] == 1).astype(int).values

        return X, y, feature_cols, df_clean.index

    def train_model(self, X_train, y_train):
        """Entraîner modèle XGBoost"""
        # Calculate class weight
        n_neg = np.sum(y_train == 0)
        n_pos = np.sum(y_train == 1)
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

        params = self.xgb_params.copy()
        params['scale_pos_weight'] = scale_pos_weight

        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, verbose=False)

        return model

    def apply_filters(self, df):
        """Appliquer Quick Win filters"""
        df['sma_50'] = df['close'].rolling(50).mean()
        df['sma_200'] = df['close'].rolling(200).mean()
        df['regime_ok'] = df['sma_50'] > df['sma_200']
        df['dow_ok'] = df.index.dayofweek != 4
        df['conf_ok'] = (df['prob'] < 0.50) | (df['prob'] >= 0.53)
        return df

    def simulate_trade(self, entry_idx, df_test, entry_price):
        """Simuler trade LONG avec TP/SL"""
        tp_price = entry_price * 1.015  # +1.5%
        sl_price = entry_price * 0.9925  # -0.75%

        lookahead = min(42, len(df_test) - entry_idx - 1)
        future_data = df_test.iloc[entry_idx+1:entry_idx+1+lookahead]

        for idx, row in future_data.iterrows():
            if row['high'] >= tp_price:
                return 'TP', tp_price, idx
            elif row['low'] <= sl_price:
                return 'SL', sl_price, idx

        return None, None, None

    def backtest_phase(self, phase, df_full, model, model_name):
        """Backtester une phase avec un modèle donné"""
        # Extract test period
        df_test = df_full[(df_full.index >= phase['test_start']) &
                         (df_full.index <= phase['test_end'])].copy()

        if len(df_test) == 0:
            return None

        # Prepare features
        X, _, feature_cols, _ = self.prepare_features(df_test)

        # Predictions
        prob = model.predict_proba(X[:, :model.n_features_in_])[:, 1]
        df_test['prob'] = prob

        # Filters
        df_test = self.apply_filters(df_test)

        # Signals
        threshold = self.thresholds[self.crypto]
        df_test['signal'] = ((df_test['prob'] > threshold) &
                            df_test['regime_ok'] &
                            df_test['dow_ok'] &
                            df_test['conf_ok']).astype(int)

        # Simulate trades
        trades = []
        capital = 1000
        consecutive_losses = 0

        for i in range(len(df_test)):
            row = df_test.iloc[i]

            if pd.isna(row['prob']) or row['signal'] == 0:
                continue

            if consecutive_losses >= 3:
                continue

            entry_price = row['close']
            outcome, exit_price, exit_time = self.simulate_trade(i, df_test, entry_price)

            if outcome is None:
                continue

            pnl_pct = ((exit_price - entry_price) / entry_price) * 100
            pnl_usd = capital * (pnl_pct / 100)
            capital += pnl_usd

            consecutive_losses = consecutive_losses + 1 if outcome == 'SL' else 0

            trades.append({
                'phase': phase['name'],
                'model': model_name,
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

        if len(trades) == 0:
            return None

        df_trades = pd.DataFrame(trades)
        n_trades = len(df_trades)
        n_tp = (df_trades['outcome'] == 'TP').sum()
        n_sl = (df_trades['outcome'] == 'SL').sum()
        win_rate = (n_tp / n_trades * 100) if n_trades > 0 else 0
        roi = ((capital - 1000) / 1000) * 100

        return {
            'phase': phase['name'],
            'model': model_name,
            'n_trades': n_trades,
            'win_rate': win_rate,
            'roi': roi,
            'final_capital': capital,
            'n_tp': int(n_tp),
            'n_sl': int(n_sl),
            'trades': df_trades
        }

    def run_comparison(self):
        """Exécuter comparaison complète"""
        print(f"\n{'='*80}")
        print(f"COMPARATIVE BACKTEST: {self.crypto.upper()} ({self.mode})")
        print(f"V11 (single model) vs V11_PHASES (retrained each phase)")
        print(f"{'='*80}")

        df_full = self.load_data()
        X_all, y_all, feature_cols, timestamps = self.prepare_features(df_full)

        # V11: Train une seule fois sur <2025
        print(f"\n{'='*80}")
        print("V11 APPROACH: Single model trained on <2025")
        print(f"{'='*80}")

        train_mask = timestamps < '2025-01-01'
        X_train_v11 = X_all[train_mask]
        y_train_v11 = y_all[train_mask]

        print(f"Training V11 model on {len(X_train_v11)} samples (2018-2024)...")
        model_v11 = self.train_model(X_train_v11, y_train_v11)
        print(f"V11 model trained: {model_v11.n_features_in_} features")

        # Test V11 sur chaque phase
        for phase in self.phases:
            print(f"\nV11 - Testing on {phase['name']}...")
            result = self.backtest_phase(phase, df_full, model_v11, 'v11')
            if result:
                self.results['v11'].append(result)
                print(f"  Trades: {result['n_trades']}, WR: {result['win_rate']:.2f}%, ROI: {result['roi']:+.2f}%")

        # V11_PHASES: Re-train pour chaque phase
        print(f"\n{'='*80}")
        print("V11_PHASES APPROACH: Model retrained for each phase")
        print(f"{'='*80}")

        for phase in self.phases:
            print(f"\nPhase: {phase['name']}")

            # Train sur période spécifique de cette phase
            train_mask_phase = (timestamps >= phase['train_start']) & (timestamps <= phase['train_end'])
            X_train_phase = X_all[train_mask_phase]
            y_train_phase = y_all[train_mask_phase]

            print(f"  Training on {len(X_train_phase)} samples ({phase['train_start']} to {phase['train_end']})...")
            model_phase = self.train_model(X_train_phase, y_train_phase)

            print(f"  Testing on {phase['test_start']} to {phase['test_end']}...")
            result = self.backtest_phase(phase, df_full, model_phase, 'v11_phases')
            if result:
                self.results['v11_phases'].append(result)
                print(f"  Trades: {result['n_trades']}, WR: {result['win_rate']:.2f}%, ROI: {result['roi']:+.2f}%")

        # Générer comparaison
        self.generate_comparison()

    def generate_comparison(self):
        """Générer tableaux et graphiques de comparaison"""
        print(f"\n{'='*80}")
        print(f"COMPARISON RESULTS: {self.crypto.upper()}")
        print(f"{'='*80}")

        # Tableau comparatif
        print(f"\n{'Phase':<20} {'Model':<12} {'Trades':<10} {'Win Rate':<12} {'ROI':<10} {'Capital':<12}")
        print("-" * 90)

        for phase in self.phases:
            phase_name = phase['name']

            # V11 result
            v11_result = next((r for r in self.results['v11'] if r['phase'] == phase_name), None)
            if v11_result:
                print(f"{phase_name:<20} {'V11':<12} {v11_result['n_trades']:<10} "
                      f"{v11_result['win_rate']:>6.2f}%    {v11_result['roi']:>+6.2f}%  "
                      f"${v11_result['final_capital']:>8.2f}")

            # V11_PHASES result
            phases_result = next((r for r in self.results['v11_phases'] if r['phase'] == phase_name), None)
            if phases_result:
                print(f"{'':<20} {'V11_PHASES':<12} {phases_result['n_trades']:<10} "
                      f"{phases_result['win_rate']:>6.2f}%    {phases_result['roi']:>+6.2f}%  "
                      f"${phases_result['final_capital']:>8.2f}")

            print()

        # Moyennes
        print("-" * 90)

        v11_avg_wr = np.mean([r['win_rate'] for r in self.results['v11']])
        v11_avg_roi = np.mean([r['roi'] for r in self.results['v11']])
        v11_total_trades = sum([r['n_trades'] for r in self.results['v11']])

        phases_avg_wr = np.mean([r['win_rate'] for r in self.results['v11_phases']])
        phases_avg_roi = np.mean([r['roi'] for r in self.results['v11_phases']])
        phases_total_trades = sum([r['n_trades'] for r in self.results['v11_phases']])

        print(f"{'AVERAGE':<20} {'V11':<12} {v11_total_trades:<10} {v11_avg_wr:>6.2f}%    {v11_avg_roi:>+6.2f}%")
        print(f"{'AVERAGE':<20} {'V11_PHASES':<12} {phases_total_trades:<10} {phases_avg_wr:>6.2f}%    {phases_avg_roi:>+6.2f}%")
        print(f"{'='*90}")

        # Déterminer le gagnant
        print(f"\n{'='*80}")
        if v11_avg_wr > phases_avg_wr:
            print(f"WINNER: V11 (single model) - {v11_avg_wr:.2f}% vs {phases_avg_wr:.2f}%")
        elif phases_avg_wr > v11_avg_wr:
            print(f"WINNER: V11_PHASES (retrained) - {phases_avg_wr:.2f}% vs {v11_avg_wr:.2f}%")
        else:
            print(f"TIE: Both models have same win rate {v11_avg_wr:.2f}%")
        print(f"{'='*80}")

        # Sauvegarder CSV
        self.save_results()

        # Graphiques
        self.generate_charts()

    def save_results(self):
        """Sauvegarder résultats en CSV"""
        # Combine tous les résultats
        all_results = []
        for r in self.results['v11']:
            all_results.append({
                'phase': r['phase'],
                'crypto': self.crypto,
                'mode': self.mode,
                'model': 'v11',
                'n_trades': r['n_trades'],
                'win_rate': r['win_rate'],
                'roi': r['roi'],
                'final_capital': r['final_capital'],
                'n_tp': r['n_tp'],
                'n_sl': r['n_sl']
            })

        for r in self.results['v11_phases']:
            all_results.append({
                'phase': r['phase'],
                'crypto': self.crypto,
                'mode': self.mode,
                'model': 'v11_phases',
                'n_trades': r['n_trades'],
                'win_rate': r['win_rate'],
                'roi': r['roi'],
                'final_capital': r['final_capital'],
                'n_tp': r['n_tp'],
                'n_sl': r['n_sl']
            })

        df_comparison = pd.DataFrame(all_results)
        comparison_file = self.results_dir / f'{self.crypto}_{self.mode}_comparison.csv'
        df_comparison.to_csv(comparison_file, index=False)
        print(f"\nComparison saved: {comparison_file.name}")

    def generate_charts(self):
        """Générer graphiques de comparaison"""
        phases_names = [p['name'] for p in self.phases]

        # Extract metrics
        v11_wr = [next((r['win_rate'] for r in self.results['v11'] if r['phase'] == p), 0) for p in phases_names]
        phases_wr = [next((r['win_rate'] for r in self.results['v11_phases'] if r['phase'] == p), 0) for p in phases_names]

        v11_roi = [next((r['roi'] for r in self.results['v11'] if r['phase'] == p), 0) for p in phases_names]
        phases_roi = [next((r['roi'] for r in self.results['v11_phases'] if r['phase'] == p), 0) for p in phases_names]

        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Win Rate comparison
        x = np.arange(len(phases_names))
        width = 0.35

        ax1.bar(x - width/2, v11_wr, width, label='V11 (single)', color='blue', alpha=0.7)
        ax1.bar(x + width/2, phases_wr, width, label='V11_PHASES (retrained)', color='green', alpha=0.7)
        ax1.set_xlabel('Phase', fontsize=12)
        ax1.set_ylabel('Win Rate (%)', fontsize=12)
        ax1.set_title(f'{self.crypto.upper()} - Win Rate Comparison', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(phases_names, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # ROI comparison
        ax2.bar(x - width/2, v11_roi, width, label='V11 (single)', color='blue', alpha=0.7)
        ax2.bar(x + width/2, phases_roi, width, label='V11_PHASES (retrained)', color='green', alpha=0.7)
        ax2.set_xlabel('Phase', fontsize=12)
        ax2.set_ylabel('ROI (%)', fontsize=12)
        ax2.set_title(f'{self.crypto.upper()} - ROI Comparison', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(phases_names, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        chart_file = self.results_dir / f'{self.crypto}_{self.mode}_comparison_chart.png'
        plt.savefig(chart_file, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Chart saved: {chart_file.name}")


def run_all_cryptos():
    """Exécuter pour toutes les cryptos"""
    configs = {
        'btc': 'baseline',
        'eth': 'top50',
        'sol': 'optuna'
    }

    print("="*80)
    print("COMPARATIVE BACKTEST: V11 vs V11_PHASES")
    print("="*80)
    print("\nComparing:")
    print("  V11: Single model trained on 2018-2024")
    print("  V11_PHASES: Model retrained for each phase")

    all_results = []

    for crypto, mode in configs.items():
        comparison = ComparativeWalkForward(crypto, mode)
        comparison.run_comparison()

        # Collect for global summary
        for r in comparison.results['v11']:
            all_results.append({**r, 'crypto': crypto, 'mode': mode})
        for r in comparison.results['v11_phases']:
            all_results.append({**r, 'crypto': crypto, 'mode': mode})

    # Global summary
    print(f"\n{'='*80}")
    print("GLOBAL SUMMARY - ALL CRYPTOS")
    print(f"{'='*80}")

    df_global = pd.DataFrame([{
        'phase': r['phase'],
        'crypto': r['crypto'],
        'mode': r['mode'],
        'model': r['model'],
        'n_trades': r['n_trades'],
        'win_rate': r['win_rate'],
        'roi': r['roi']
    } for r in all_results])

    results_dir = Path(__file__).parent.parent / 'results' / 'compare_v11_phases'
    global_file = results_dir / 'GLOBAL_COMPARISON.csv'
    df_global.to_csv(global_file, index=False)

    print("\nGlobal comparison saved")
    print(f"\nResults in: {results_dir}")
    print("="*80)


if __name__ == '__main__':
    run_all_cryptos()
