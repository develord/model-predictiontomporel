"""
Script de vérification CRITIQUE des labels avant entraînement
=============================================================
Vérifie la distribution des triple barrier labels pour détecter
les déséquilibres qui pourraient ruiner le modèle.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


def check_label_distribution(crypto):
    """Vérifie la distribution des labels pour un crypto"""

    print(f"\n{'='*80}")
    print(f"{crypto.upper()} - ANALYSE DISTRIBUTION LABELS")
    print('='*80)

    cache_file = Path(__file__).parent.parent / 'data' / 'cache' / f'{crypto}_multi_tf_merged.csv'

    if not cache_file.exists():
        print(f"❌ Fichier non trouvé: {cache_file}")
        print(f"   → Exécuter d'abord: python features/multi_tf_pipeline.py")
        return None

    df = pd.read_csv(cache_file, index_col=0)

    print(f"\nDataset: {len(df)} rows, {len(df.columns)} columns")

    # === TRIPLE BARRIER LABELS ===
    print(f"\n{'─'*80}")
    print("TRIPLE BARRIER LABELS (Regression Target)")
    print('─'*80)

    if 'triple_barrier_label' in df.columns:
        tb_labels = df['triple_barrier_label'].dropna()

        if len(tb_labels) == 0:
            print("❌ ERREUR: Aucun label triple_barrier généré!")
            return None

        # Distribution
        dist = tb_labels.value_counts().sort_index()
        dist_pct = tb_labels.value_counts(normalize=True).sort_index()

        print(f"\nDistribution absolue:")
        label_names = {-1: 'SL (Stop Loss)', 0: 'Timeout', 1: 'TP (Take Profit)'}
        for label in sorted(dist.index):
            name = label_names.get(int(label), f'Label {label}')
            count = dist[label]
            pct = dist_pct[label] * 100
            print(f"  {name:20s}: {count:6d} ({pct:5.1f}%)")

        print(f"\nTotal valid labels: {len(tb_labels):,}")
        print(f"NaN labels: {df['triple_barrier_label'].isna().sum():,}")

        # === DIAGNOSTIC ===
        print(f"\n{'─'*40}")
        print("DIAGNOSTIC")
        print('─'*40)

        issues = []
        warnings = []

        # Vérifier si 3 classes
        if len(dist) < 3:
            issues.append(f"⚠️  Seulement {len(dist)} classes (attendu: 3)")

        # Vérifier déséquilibre
        if len(dist_pct) == 3:
            min_pct = dist_pct.min()
            max_pct = dist_pct.max()

            if min_pct < 0.10:
                issues.append(f"❌ Classe minoritaire: {min_pct*100:.1f}% (< 10% → Très déséquilibré!)")
            elif min_pct < 0.15:
                warnings.append(f"⚠️  Classe minoritaire: {min_pct*100:.1f}% (< 15% → Déséquilibré)")

            if max_pct > 0.70:
                issues.append(f"❌ Classe majoritaire: {max_pct*100:.1f}% (> 70% → Très déséquilibré!)")
            elif max_pct > 0.60:
                warnings.append(f"⚠️  Classe majoritaire: {max_pct*100:.1f}% (> 60% → Déséquilibré)")

            # Ratio TP/SL
            if 1 in dist.index and -1 in dist.index:
                tp_count = dist[1]
                sl_count = dist[-1]
                ratio = tp_count / sl_count if sl_count > 0 else 0

                if ratio < 0.5 or ratio > 2.0:
                    warnings.append(f"⚠️  Ratio TP/SL déséquilibré: {ratio:.2f} (optimal: 0.7-1.4)")

        # Afficher diagnostics
        if not issues and not warnings:
            print("✅ Distribution EXCELLENTE - Pas de problème détecté")
        else:
            if issues:
                print("\nPROBLÈMES CRITIQUES:")
                for issue in issues:
                    print(f"  {issue}")
            if warnings:
                print("\nAvertissements:")
                for warning in warnings:
                    print(f"  {warning}")

        # === RECOMMANDATIONS ===
        if issues or warnings:
            print(f"\n{'─'*40}")
            print("RECOMMANDATIONS")
            print('─'*40)

            if len(dist_pct) == 3:
                min_pct = dist_pct.min()
                max_pct = dist_pct.max()

                if min_pct < 0.15:
                    print("\n1. Ajuster class_weight dans XGBoost:")
                    print("   scale_pos_weight = (majority_count / minority_count)")
                    print("\n2. OU ajuster seuils TP/SL:")
                    if dist_pct[1] < 0.15:  # TP minoritaire
                        print("   → Baisser TP (ex: 1.2% au lieu de 1.5%)")
                    if dist_pct[-1] < 0.15:  # SL minoritaire
                        print("   → Baisser SL (ex: 0.5% au lieu de 0.75%)")
                    if dist_pct.get(0, 0) < 0.15:  # Timeout minoritaire
                        print("   → Réduire lookahead_candles (ex: 5 au lieu de 7)")

        return {
            'crypto': crypto,
            'total_labels': len(tb_labels),
            'distribution': dist.to_dict(),
            'distribution_pct': dist_pct.to_dict(),
            'has_issues': len(issues) > 0,
            'has_warnings': len(warnings) > 0,
            'issues': issues,
            'warnings': warnings
        }

    else:
        print("❌ Colonne 'triple_barrier_label' non trouvée!")
        print("   → Vérifier que labels.py génère bien les labels")
        return None


def check_classification_labels(crypto):
    """Vérifie aussi les labels de classification"""

    print(f"\n{'─'*80}")
    print("CLASSIFICATION LABELS (BUY/SELL/HOLD)")
    print('─'*80)

    cache_file = Path(__file__).parent.parent / 'data' / 'cache' / f'{crypto}_multi_tf_merged.csv'
    df = pd.read_csv(cache_file, index_col=0)

    if 'label_class' in df.columns:
        class_labels = df['label_class'].dropna()
        dist = class_labels.value_counts()
        dist_pct = class_labels.value_counts(normalize=True)

        print(f"\nDistribution:")
        for label in ['BUY', 'HOLD', 'SELL']:
            if label in dist.index:
                count = dist[label]
                pct = dist_pct[label] * 100
                print(f"  {label:6s}: {count:6d} ({pct:5.1f}%)")

        print(f"\nTotal: {len(class_labels):,}")


if __name__ == '__main__':
    print("="*80)
    print("V10 LABEL DISTRIBUTION CHECK")
    print("="*80)

    cryptos = ['btc', 'eth', 'sol']
    results = {}

    for crypto in cryptos:
        result = check_label_distribution(crypto)
        if result:
            results[crypto] = result

        # Aussi vérifier labels classification
        check_classification_labels(crypto)

    # === SUMMARY ===
    print(f"\n{'='*80}")
    print("SUMMARY")
    print('='*80)

    all_ok = True
    for crypto, result in results.items():
        status = "✅ OK" if not result['has_issues'] else "❌ PROBLÈMES"
        if result['has_warnings'] and not result['has_issues']:
            status = "⚠️  Avertissements"

        print(f"\n{crypto.upper()}: {status}")
        if result['has_issues']:
            all_ok = False
            print("  → Correction REQUISE avant entraînement")
        elif result['has_warnings']:
            print("  → Utilisable mais sous-optimal")

    if all_ok:
        print(f"\n{'='*80}")
        print("✅ TOUS LES CRYPTOS OK - Prêt pour entraînement!")
        print('='*80)
    else:
        print(f"\n{'='*80}")
        print("❌ CORRECTIONS REQUISES - Voir recommandations ci-dessus")
        print('='*80)
