# Plan d'Action - Optimisation Win Rate V12

## Date: 2026-03-24
## Objectif: Passer le Win Rate de ~45-51% vers 55%+ tout en gardant le ROI V12

---

## Diagnostic

| Crypto | V11 WR | V12 WR | V11 ROI | V12 ROI |
|--------|--------|--------|---------|---------|
| BTC    | 55.6%  | 50.7%  | +10.15% | +12.13% |
| ETH    | 61.4%  | 52.1%  | +14.35% | +18.35% |
| SOL    | 76.0%  | 44.7%  | +25.21% | +30.20% |

**Probleme principal**: Le modele est entraine sur des labels V11 fixes (TP=1.5%, SL=0.75%) mais execute avec des TP/SL ATR dynamiques. Ce mismatch tue le win rate.

**Autres problemes**:
- Seuil de confiance trop bas (0.35) → trop de signaux faibles
- XGBoost optimise pour AUC, pas pour la precision (= win rate)
- LSTM sous-exploite (rank #87-221 sur 242 features, importance negligeable)

---

## Axes d'Optimisation

### AXE 1 - Retrain sur Labels ATR Dynamiques
**Priorite: CRITIQUE**

Le modele doit predire ce qu'il va reellement trader.
- Generer des labels avec `apply_dynamic_triple_barrier()` utilisant les memes parametres ATR que l'execution
- BTC/ETH: tp_mult=0.40, sl_mult=0.15
- SOL: tp_mult=0.60, sl_mult=0.15
- Memes bornes min/max TP/SL et R:R min que la config V12

**Impact attendu**: +5-10% WR (alignement training/execution)

### AXE 2 - XGBoost Optimise Precision
**Priorite: HAUTE**

Changer l'objectif d'optimisation:
- `eval_metric`: `aucpr` au lieu de `auc`
- 3 configs a tester:
  - **Conservative**: max_depth=4, lr=0.03, n_est=300, gamma=3, min_child=5, colsample=0.6
  - **Balanced**: max_depth=5, lr=0.05, n_est=250, gamma=2, min_child=3, colsample=0.7
  - **Aggressive**: max_depth=6, lr=0.05, n_est=200, gamma=2, min_child=1, colsample=0.8
- Tester chaque config sur P1 + P2

**Impact attendu**: +2-5% WR (moins de faux positifs)

### AXE 3 - Grid Search Seuil de Confiance
**Priorite: HAUTE**

Le seuil actuel (0.35) est trop permissif:
- Tester: [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
- Contrainte: minimum 30 trades par periode
- Contrainte: ROI > 0
- Ranking: WR desc, puis ROI desc

**Impact attendu**: +3-8% WR (filtrage signaux faibles)

### AXE 4 - LSTM comme Filtre Veto
**Priorite: MOYENNE**

Au lieu d'utiliser le LSTM comme simple feature (importance faible), l'utiliser comme filtre dur:
- Si XGBoost dit TRADE mais LSTM dit NON → SKIP
- Tester seuils LSTM veto: [0.45, 0.50, 0.55]
- Le LSTM doit etre d'accord pour trader

**Impact attendu**: +2-5% WR (double confirmation)

---

## Pipeline d'Execution

```
1. Charger donnees multi-TF merged
2. Generer labels ATR dynamiques (AXE 1)
3. Construire features LSTM avec cutoff temporel
4. Pour chaque config XGB (AXE 2):
   a. Entrainer sur labels dynamiques
   b. Pour chaque seuil confiance (AXE 3):
      - Backtest sans LSTM veto
      - Backtest avec LSTM veto a 3 niveaux (AXE 4)
5. Walk-forward validation P1 + P2
6. Selectionner meilleure config consistante
7. Tableau comparatif final: V11 vs V12 current vs V12 optimise
```

**Total combos par crypto par periode**: 3 configs × 7 seuils × 4 veto options = 84

---

## Criteres de Succes

| Metrique | Minimum | Cible |
|----------|---------|-------|
| Win Rate moyen (P1+P2) | > 55% | > 60% |
| ROI moyen (P1+P2) | > 5% | > 10% |
| Profit Factor | > 1.5 | > 2.0 |
| Min trades/periode | 30 | 50+ |
| Consistance P1/P2 WR | < 10% ecart | < 5% ecart |

---

## Script

Le script d'optimisation est pret: `v12/optimization/optimize_v12_winrate.py`

### Dependances
```
pip install pandas numpy scikit-learn xgboost joblib requests torch
```

### Execution
```bash
python v12/optimization/optimize_v12_winrate.py
```

### Output
- Console: tableaux comparatifs detailles
- Fichier: `v12/results/optimize_v12_winrate_results.json`

---

## Prochaines Etapes (apres optimisation)

1. **Retrain final** avec la meilleure config trouvee
2. **Sauvegarder** nouveaux modeles dans `v12/models/`
3. **Mettre a jour** `v12/config/v12_config.json` avec les nouveaux seuils
4. **Backtest final** Q1 2025 pour validation
5. **Comparer** V11 vs V12 current vs V12 optimise (tableau final)
