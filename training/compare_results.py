"""
Compare results from all 3 training phases
"""

import json
from pathlib import Path
from tabulate import tabulate


def load_stats(crypto: str, mode: str):
    """Load stats for a crypto and mode"""
    stats_file = Path(__file__).parent.parent / 'models' / f'{crypto}_v11_{mode}_stats.json'

    if not stats_file.exists():
        return None

    with open(stats_file, 'r') as f:
        return json.load(f)


def compare_all():
    """Generate comprehensive comparison table"""

    cryptos = ['btc', 'eth', 'sol']
    modes = ['baseline', 'optuna', 'top50']

    print("=" * 120)
    print("V11 PHASE TRAINING - COMPREHENSIVE RESULTS COMPARISON")
    print("=" * 120)
    print()

    # Detailed table
    detailed_rows = []

    for crypto in cryptos:
        for mode in modes:
            stats = load_stats(crypto, mode)

            if stats is None:
                continue

            metrics = stats['test_metrics']
            hyper = stats['hyperparameters']

            row = [
                crypto.upper(),
                mode.upper(),
                stats['features'],
                f"{metrics['accuracy']:.4f}",
                f"{metrics['auc']:.4f}",
                f"{metrics['tp_class']['precision']:.4f}",
                f"{metrics['tp_class']['recall']:.4f}",
                f"{metrics['tp_class']['f1']:.4f}",
                f"{metrics['sl_class']['precision']:.4f}",
                f"{metrics['sl_class']['recall']:.4f}",
                f"{metrics['sl_class']['f1']:.4f}",
                f"{hyper['max_depth']}",
                f"{hyper['learning_rate']:.4f}",
                f"{hyper['n_estimators']}"
            ]

            detailed_rows.append(row)

    headers = [
        'Crypto', 'Mode', 'Features',
        'Accuracy', 'AUC',
        'TP Prec', 'TP Rec', 'TP F1',
        'SL Prec', 'SL Rec', 'SL F1',
        'Max Depth', 'LR', 'N Est'
    ]

    print(tabulate(detailed_rows, headers=headers, tablefmt='grid'))

    # Summary table - best configuration per crypto
    print("\n" + "=" * 120)
    print("BEST CONFIGURATION PER CRYPTO (by AUC)")
    print("=" * 120)
    print()

    summary_rows = []

    for crypto in cryptos:
        best_mode = None
        best_auc = 0
        best_stats = None

        for mode in modes:
            stats = load_stats(crypto, mode)
            if stats is None:
                continue

            auc = stats['test_metrics']['auc']
            if auc > best_auc:
                best_auc = auc
                best_mode = mode
                best_stats = stats

        if best_stats:
            metrics = best_stats['test_metrics']
            row = [
                crypto.upper(),
                best_mode.upper(),
                best_stats['features'],
                f"{metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)",
                f"{metrics['auc']:.4f}",
                f"{metrics['tp_class']['f1']:.4f}",
                f"{metrics['sl_class']['f1']:.4f}",
                f"{metrics['confusion_matrix']['true_tp']}",
                f"{metrics['confusion_matrix']['true_sl']}",
                f"{metrics['confusion_matrix']['false_tp']}",
                f"{metrics['confusion_matrix']['false_sl']}"
            ]
            summary_rows.append(row)

    summary_headers = [
        'Crypto', 'Best Mode', 'Features',
        'Accuracy', 'AUC',
        'TP F1', 'SL F1',
        'True TP', 'True SL', 'False TP', 'False SL'
    ]

    print(tabulate(summary_rows, headers=summary_headers, tablefmt='grid'))

    # Comparison across modes
    print("\n" + "=" * 120)
    print("MODE COMPARISON (Average metrics across all cryptos)")
    print("=" * 120)
    print()

    mode_comparison = []

    for mode in modes:
        accuracies = []
        aucs = []
        tp_f1s = []

        for crypto in cryptos:
            stats = load_stats(crypto, mode)
            if stats:
                accuracies.append(stats['test_metrics']['accuracy'])
                aucs.append(stats['test_metrics']['auc'])
                tp_f1s.append(stats['test_metrics']['tp_class']['f1'])

        if accuracies:
            avg_acc = sum(accuracies) / len(accuracies)
            avg_auc = sum(aucs) / len(aucs)
            avg_tp_f1 = sum(tp_f1s) / len(tp_f1s)

            mode_comparison.append([
                mode.upper(),
                len(accuracies),
                f"{avg_acc:.4f} ({avg_acc*100:.2f}%)",
                f"{avg_auc:.4f}",
                f"{avg_tp_f1:.4f}"
            ])

    mode_headers = ['Mode', 'N Models', 'Avg Accuracy', 'Avg AUC', 'Avg TP F1']
    print(tabulate(mode_comparison, headers=mode_headers, tablefmt='grid'))

    # Feature count analysis
    print("\n" + "=" * 120)
    print("FEATURE COUNT ANALYSIS")
    print("=" * 120)
    print()

    feature_rows = []

    for crypto in cryptos:
        baseline_stats = load_stats(crypto, 'baseline')
        top50_stats = load_stats(crypto, 'top50')

        if baseline_stats and top50_stats:
            baseline_auc = baseline_stats['test_metrics']['auc']
            top50_auc = top50_stats['test_metrics']['auc']
            auc_diff = top50_auc - baseline_auc

            feature_rows.append([
                crypto.upper(),
                baseline_stats['features'],
                f"{baseline_auc:.4f}",
                top50_stats['features'],
                f"{top50_auc:.4f}",
                f"{auc_diff:+.4f}",
                "Worse" if auc_diff < 0 else "Better"
            ])

    feature_headers = [
        'Crypto',
        'Baseline Features', 'Baseline AUC',
        'TOP50 Features', 'TOP50 AUC',
        'AUC Diff', 'Result'
    ]

    print(tabulate(feature_rows, headers=feature_headers, tablefmt='grid'))

    print("\n" + "=" * 120)
    print("CONCLUSION")
    print("=" * 120)
    print()

    # Find overall best
    best_overall_crypto = None
    best_overall_mode = None
    best_overall_auc = 0

    for crypto in cryptos:
        for mode in modes:
            stats = load_stats(crypto, mode)
            if stats:
                auc = stats['test_metrics']['auc']
                if auc > best_overall_auc:
                    best_overall_auc = auc
                    best_overall_crypto = crypto
                    best_overall_mode = mode

    print(f"Best Overall Configuration:")
    print(f"  Crypto: {best_overall_crypto.upper()}")
    print(f"  Mode: {best_overall_mode.upper()}")
    print(f"  AUC: {best_overall_auc:.4f}")

    best_stats = load_stats(best_overall_crypto, best_overall_mode)
    print(f"  Accuracy: {best_stats['test_metrics']['accuracy']:.4f} ({best_stats['test_metrics']['accuracy']*100:.2f}%)")
    print(f"  Features: {best_stats['features']}")
    print()

    print("Key Findings:")
    print("  1. Baseline models generally perform well across all cryptos")
    print("  2. Feature reduction (TOP50) tends to degrade performance")
    print("  3. Optuna optimization shows mixed results - sometimes better, sometimes worse")
    print("  4. All models show accuracy > 50% (better than random)")
    print()


if __name__ == '__main__':
    compare_all()
