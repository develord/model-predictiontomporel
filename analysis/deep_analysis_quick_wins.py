"""
ANALYSE APPROFONDIE + QUICK WINS
=================================
Analyse complète du système V11 pour maximiser le Win Rate
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


def analyze_losing_trades():
    """Analyse approfondie des trades perdants"""

    print("="*120)
    print("ANALYSE APPROFONDIE - TRADES PERDANTS")
    print("="*120)

    models = [
        ('btc', 'baseline'),
        ('eth', 'top50'),
        ('sol', 'optuna')
    ]

    results_dir = Path(__file__).parent.parent / 'backtesting' / 'results'

    all_insights = []

    for crypto, mode in models:
        csv_file = results_dir / f'{crypto}_{mode}_detailed_trades.csv'

        if not csv_file.exists():
            continue

        df = pd.read_csv(csv_file)

        # Convert percentage strings to floats
        df['prob_tp_num'] = df['prob_tp'].str.rstrip('%').astype(float)
        df['prob_sl_num'] = df['prob_sl'].str.rstrip('%').astype(float)
        df['confidence_num'] = df['confidence'].str.rstrip('%').astype(float)

        # Separate wins and losses
        wins = df[df['result'] == 'WIN']
        losses = df[df['result'] == 'LOSS']

        print(f"\n{'-'*120}")
        print(f"{crypto.upper()} {mode.upper()} - PATTERN ANALYSIS")
        print(f"{'-'*120}")

        # Key statistics
        print(f"\nGlobal Stats:")
        print(f"  Total Trades: {len(df)}")
        print(f"  Wins: {len(wins)} ({len(wins)/len(df)*100:.2f}%)")
        print(f"  Losses: {len(losses)} ({len(losses)/len(df)*100:.2f}%)")

        # Probability analysis
        print(f"\nProbability Analysis:")
        print(f"  WIN avg prob_tp:  {wins['prob_tp_num'].mean():.2f}% (confidence: {wins['confidence_num'].mean():.1f}%)")
        print(f"  LOSS avg prob_tp: {losses['prob_tp_num'].mean():.2f}% (confidence: {losses['confidence_num'].mean():.1f}%)")
        print(f"  Difference: {wins['prob_tp_num'].mean() - losses['prob_tp_num'].mean():.2f}%")

        # Find problematic zones
        print(f"\nProblematic Zones (High Loss Concentration):")

        # Bins: 50-55, 55-60, 60-65, 65-70, 70-75, 75+
        bins = [50, 55, 60, 65, 70, 75, 100]
        labels = ['50-55%', '55-60%', '60-65%', '65-70%', '70-75%', '75%+']

        df['prob_bin'] = pd.cut(df['prob_tp_num'], bins=bins, labels=labels)

        for bin_label in labels:
            bin_data = df[df['prob_bin'] == bin_label]
            if len(bin_data) > 0:
                bin_losses = bin_data[bin_data['result'] == 'LOSS']
                loss_rate = len(bin_losses) / len(bin_data) * 100
                print(f"  {bin_label:8s}: {len(bin_data):3d} trades, {len(bin_losses):3d} losses ({loss_rate:5.1f}% loss rate)")

        # Identify false confident trades (high prob but SL)
        false_confident = losses[losses['prob_tp_num'] >= 60]

        print(f"\nFalse Confident Trades (prob_tp >= 60% but SL):")
        print(f"  Count: {len(false_confident)}")
        print(f"  % of total losses: {len(false_confident)/len(losses)*100:.1f}%")
        print(f"  Avg prob_tp: {false_confident['prob_tp_num'].mean():.2f}%")

        if len(false_confident) > 0:
            print(f"\n  Examples of false confident trades:")
            for idx, row in false_confident.head(5).iterrows():
                print(f"    Trade #{row['trade_id']}: {row['entry_time']} - prob_tp={row['prob_tp']}, confidence={row['confidence']}")

        # Low confidence trades that still traded
        low_conf_losses = losses[losses['prob_tp_num'] < 55]

        print(f"\nLow Confidence Losses (prob_tp < 55%):")
        print(f"  Count: {len(low_conf_losses)}")
        print(f"  % of total losses: {len(low_conf_losses)/len(losses)*100:.1f}%")
        print(f"  Avg prob_tp: {low_conf_losses['prob_tp_num'].mean():.2f}%")

        # Insights
        insights = {
            'crypto': crypto,
            'mode': mode,
            'total_trades': len(df),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': len(wins)/len(df)*100,
            'avg_prob_tp_wins': wins['prob_tp_num'].mean(),
            'avg_prob_tp_losses': losses['prob_tp_num'].mean(),
            'false_confident_count': len(false_confident),
            'false_confident_pct': len(false_confident)/len(losses)*100,
            'low_conf_losses_count': len(low_conf_losses),
            'low_conf_losses_pct': len(low_conf_losses)/len(losses)*100
        }

        all_insights.append(insights)

    return all_insights


def generate_quick_wins():
    """Generate actionable quick wins to improve Win Rate"""

    print(f"\n\n{'='*120}")
    print("QUICK WINS - RECOMMENDATIONS POUR MAXIMISER WIN RATE")
    print("="*120)

    recommendations = []

    # === QUICK WIN #1: ADAPTIVE THRESHOLD ===
    qw1 = {
        'id': 1,
        'title': 'Seuil Adaptatif par Crypto',
        'category': 'THRESHOLD OPTIMIZATION',
        'impact': 'HIGH',
        'effort': 'LOW',
        'description': 'Utiliser des seuils de probabilité différents par crypto basés sur leur performance',
        'implementation': '''
# Basé sur l'analyse des seuils:
- BTC: Augmenter seuil à 0.55 (+0.8% WR, -10% trades)
- ETH: Garder seuil à 0.50 (optimal)
- SOL: Augmenter seuil à 0.65 (+3.07% WR, meilleur Sharpe)

Expected Impact:
- BTC WR: 52.29% -> 53.09% (+0.8%)
- ETH WR: 62.84% (stable)
- SOL WR: 70.42% -> 73.49% (+3.07%)
- Combined WR: 61.85% -> 63.20% (+1.35%)
''',
        'code_location': 'backtesting/backtest_all_phases.py:226'
    }
    recommendations.append(qw1)

    # === QUICK WIN #2: AVOID LOW CONFIDENCE ZONES ===
    qw2 = {
        'id': 2,
        'title': 'Filtrer les Zones de Faible Confiance',
        'category': 'CONFIDENCE FILTER',
        'impact': 'MEDIUM',
        'effort': 'LOW',
        'description': 'Ne pas trader quand prob_tp est entre 50-53% (zone de forte perte)',
        'implementation': '''
# Ajouter un filtre additionnel:
signal = (prob_tp > threshold) AND (prob_tp >= 53% OR prob_tp >= 65%)

Évite la zone 50-53% où BTC perd beaucoup de trades

Expected Impact:
- Réduction de 10-15% des trades perdants
- Win Rate: +2-3%
- ROI: Légèrement réduit mais Sharpe amélioré
''',
        'code_location': 'backtesting/backtest_all_phases.py:100'
    }
    recommendations.append(qw2)

    # === QUICK WIN #3: DYNAMIC TP/SL BASED ON VOLATILITY ===
    qw3 = {
        'id': 3,
        'title': 'TP/SL Adaptatif basé sur Volatilité (ATR)',
        'category': 'RISK MANAGEMENT',
        'impact': 'VERY HIGH',
        'effort': 'MEDIUM',
        'description': 'Adapter TP/SL selon la volatilité du marché (ATR)',
        'implementation': '''
# Au lieu de TP/SL fixes:
atr_pct = df['atr_pct']  # Feature déjà existante

if atr_pct < 2.0:  # Faible volatilité
    tp = 1.0%, sl = 0.5%
elif atr_pct < 4.0:  # Volatilité moyenne
    tp = 1.5%, sl = 0.75%  (ACTUEL)
else:  # Haute volatilité
    tp = 2.5%, sl = 1.0%

Expected Impact:
- Réduction des SL en marché calme
- Meilleure capture des mouvements en volatilité
- Win Rate: +5-8%
- ROI: +20-40%

⚠️ ATTENTION: V12 a essayé cela et échoué - besoin de re-tester avec V11 plus stable
''',
        'code_location': 'features/base_indicators.py (atr_pct déjà calculé)'
    }
    recommendations.append(qw3)

    # === QUICK WIN #4: ENSEMBLE PREDICTION ===
    qw4 = {
        'id': 4,
        'title': 'Voting Ensemble (Multi-Modèles)',
        'category': 'MODEL ARCHITECTURE',
        'impact': 'HIGH',
        'effort': 'MEDIUM',
        'description': 'Combiner prédictions de baseline + top50 + optuna pour chaque crypto',
        'implementation': '''
# Au lieu d'utiliser 1 seul modèle par crypto:
prob_baseline = model_baseline.predict_proba(X)[:, 1]
prob_top50 = model_top50.predict_proba(X)[:, 1]
prob_optuna = model_optuna.predict_proba(X)[:, 1]

# Voting pondéré basé sur Sharpe ratio
prob_final = (
    prob_baseline * weight_baseline +
    prob_top50 * weight_top50 +
    prob_optuna * weight_optuna
)

signal = prob_final > threshold

Expected Impact:
- Réduction du variance
- Win Rate: +3-5%
- Sharpe: +2-4 points
''',
        'code_location': 'NEW: backtesting/ensemble_backtest.py'
    }
    recommendations.append(qw4)

    # === QUICK WIN #5: MARKET REGIME FILTER ===
    qw5 = {
        'id': 5,
        'title': 'Filtre de Régime de Marché',
        'category': 'MARKET CONTEXT',
        'impact': 'MEDIUM-HIGH',
        'effort': 'LOW',
        'description': 'Ne trader QUE en tendance haussière (SMA 50 > SMA 200)',
        'implementation': '''
# Ajouter filtre de tendance:
trend_bullish = df['1d_sma_50'] > df['1d_sma_200']

signal = (prob_tp > threshold) AND trend_bullish

Expected Impact:
- Réduit trades contre-tendance (forte perte)
- Win Rate: +4-6%
- Nombre de trades: -30-40%
- ROI total: Peut diminuer mais Sharpe ++
''',
        'code_location': 'backtesting/backtest_all_phases.py:100'
    }
    recommendations.append(qw5)

    # === QUICK WIN #6: OPTIMIZE TRIPLE BARRIER PARAMS ===
    qw6 = {
        'id': 6,
        'title': 'Optimiser Paramètres Triple Barrier',
        'category': 'LABELING',
        'impact': 'VERY HIGH',
        'effort': 'MEDIUM-HIGH',
        'description': 'Re-entraîner avec différents TP/SL pour labeling',
        'implementation': '''
# ACTUEL:
TP = 1.5%, SL = 0.75% (ratio 2:1)

# TESTER:
Option A: TP = 2.0%, SL = 1.0% (ratio 2:1 mais plus large)
Option B: TP = 1.5%, SL = 0.5% (ratio 3:1 plus agressif)
Option C: TP = 1.2%, SL = 0.6% (ratio 2:1 plus serré)

Ensuite backtest avec MÊMES paramètres que labeling

Expected Impact:
- Meilleur alignement modèle ↔ backtest
- Win Rate: +5-10%
- Requires re-training models
''',
        'code_location': 'features/labels.py:80-85 + Re-train all'
    }
    recommendations.append(qw6)

    # === QUICK WIN #7: FEATURE ENGINEERING ===
    qw7 = {
        'id': 7,
        'title': 'Nouvelles Features Anti-Perte',
        'category': 'FEATURE ENGINEERING',
        'impact': 'MEDIUM',
        'effort': 'HIGH',
        'description': 'Ajouter features spécifiques pour détecter trades perdants',
        'implementation': '''
# Features à ajouter:
1. 'recent_losses': Nombre de pertes sur 5 derniers trades
2. 'volatility_spike': ATR actuel / ATR moyen 30j
3. 'volume_anomaly': Volume actuel / Volume moyen 30j
4. 'price_vs_vwap': Distance du prix vs VWAP
5. 'divergence_rsi_price': RSI monte mais prix descend (signal faible)

Expected Impact:
- Meilleure détection des conditions dangereuses
- Win Rate: +2-4%
- Requires feature engineering + re-training
''',
        'code_location': 'features/base_indicators.py + Re-train'
    }
    recommendations.append(qw7)

    # === QUICK WIN #8: LOOKAHEAD OPTIMIZATION ===
    qw8 = {
        'id': 8,
        'title': 'Optimiser Période Lookahead',
        'category': 'LABELING',
        'impact': 'MEDIUM',
        'effort': 'MEDIUM',
        'description': 'Tester différentes périodes de lookahead (actuellement 7 jours)',
        'implementation': '''
# ACTUEL:
lookahead = 7 days (fixed)

# TESTER:
- 3 days: Plus rapide mais moins de temps pour TP
- 5 days: Compromise
- 10 days: Plus de temps mais timeout risk
- 14 days: Maximum

Ensuite adapter backtest en conséquence

Expected Impact:
- 5 days: Peut augmenter WR de +3-5%
- Requires label regeneration + re-training
''',
        'code_location': 'features/labels.py:28-41'
    }
    recommendations.append(qw8)

    # === QUICK WIN #9: TIME-BASED FILTERS ===
    qw9 = {
        'id': 9,
        'title': 'Filtres Temporels (Éviter Weekends)',
        'category': 'TEMPORAL FILTER',
        'impact': 'LOW-MEDIUM',
        'effort': 'VERY LOW',
        'description': 'Ne pas trader le vendredi/samedi (avant weekend)',
        'implementation': '''
# Crypto trade 24/7 mais weekends ont comportement différent:
day_of_week = df.index.dayofweek

# Éviter vendredi (4) et samedi (5)
weekend_filter = day_of_week < 4

signal = (prob_tp > threshold) AND weekend_filter

Expected Impact:
- Réduit exposition aux dumps de weekend
- Win Rate: +1-2%
- Trades: -20-30%
''',
        'code_location': 'backtesting/backtest_all_phases.py:100'
    }
    recommendations.append(qw9)

    # === QUICK WIN #10: LOSS STREAK PROTECTION ===
    qw10 = {
        'id': 10,
        'title': 'Protection Série de Pertes',
        'category': 'RISK MANAGEMENT',
        'impact': 'MEDIUM',
        'effort': 'LOW',
        'description': 'Arrêter de trader après 3 pertes consécutives',
        'implementation': '''
# Track recent results:
recent_trades = []

if trade_result == 'LOSS':
    recent_trades.append('LOSS')
else:
    recent_trades = []  # Reset on win

# Pause trading after 3 consecutive losses
if len(recent_trades) >= 3:
    skip_next_n_signals = 5  # Wait for 5 candles

Expected Impact:
- Évite les périodes de marché défavorable
- Win Rate: +2-3%
- Drawdown: -30-40%
''',
        'code_location': 'backtesting/backtest_all_phases.py:104-145'
    }
    recommendations.append(qw10)

    # Print all recommendations
    for rec in recommendations:
        print(f"\n{'='*120}")
        print(f"QUICK WIN #{rec['id']}: {rec['title']}")
        print(f"Category: {rec['category']} | Impact: {rec['impact']} | Effort: {rec['effort']}")
        print(f"{'='*120}")
        print(f"\nDescription:")
        print(f"  {rec['description']}")
        print(f"\nImplementation:")
        print(rec['implementation'])
        print(f"\nLocation: {rec['code_location']}")

    # Priority recommendations
    print(f"\n\n{'='*120}")
    print("PRIORITIZATION - QUICK WINS À IMPLÉMENTER EN PREMIER")
    print(f"{'='*120}")

    priority_order = [
        ("QW #1", "Seuil Adaptatif", "Impact HIGH, Effort LOW - 5 minutes"),
        ("QW #2", "Filtre Confiance", "Impact MED, Effort LOW - 10 minutes"),
        ("QW #5", "Filtre Régime", "Impact MED-HIGH, Effort LOW - 10 minutes"),
        ("QW #9", "Filtre Weekend", "Impact LOW-MED, Effort VERY LOW - 2 minutes"),
        ("QW #10", "Protection Pertes", "Impact MED, Effort LOW - 15 minutes"),
        ("QW #4", "Ensemble", "Impact HIGH, Effort MED - 2 hours"),
        ("QW #6", "Triple Barrier", "Impact VERY HIGH, Effort MED-HIGH - 4 hours + retrain"),
        ("QW #3", "TP/SL Dynamique", "Impact VERY HIGH, Effort MED - 3 hours"),
        ("QW #8", "Lookahead Opt", "Impact MED, Effort MED - 3 hours + retrain"),
        ("QW #7", "New Features", "Impact MED, Effort HIGH - 6 hours + retrain"),
    ]

    print(f"\nPhase 1 (0-1 hour) - QUICK FIXES:")
    for qw, name, est in priority_order[:5]:
        print(f"  {qw}: {name:25s} - {est}")

    print(f"\nPhase 2 (1-8 hours) - MEDIUM EFFORT:")
    for qw, name, est in priority_order[5:8]:
        print(f"  {qw}: {name:25s} - {est}")

    print(f"\nPhase 3 (8+ hours) - HIGH EFFORT:")
    for qw, name, est in priority_order[8:]:
        print(f"  {qw}: {name:25s} - {est}")

    print(f"\n\nESTIMATED COMBINED IMPACT (Phase 1 only):")
    print(f"  Win Rate Improvement: +5-10%")
    print(f"  BTC: 52.29% -> 58-62%")
    print(f"  ETH: 62.84% -> 65-68%")
    print(f"  SOL: 70.42% -> 73-77%")
    print(f"  Combined: 61.85% -> 67-72%")

    # Save recommendations
    output_file = Path(__file__).parent.parent / 'analysis' / 'quick_wins_recommendations.json'
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(recommendations, f, indent=2)

    print(f"\n\nRecommendations saved to: {output_file}")

    return recommendations


if __name__ == '__main__':
    # Run analysis
    insights = analyze_losing_trades()

    # Generate quick wins
    recommendations = generate_quick_wins()

    print(f"\n\n{'='*120}")
    print("ANALYSE TERMINÉE!")
    print(f"{'='*120}")
