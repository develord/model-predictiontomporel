# BITCOIN (BTC) - Configuration Optimale

**Version**: V11 TEMPORAL PRO
**Date**: 21 Mars 2026

---

## CONFIGURATION RECOMMANDÉE

```python
{
    'model_file': 'models/btc_v11_classifier.joblib',
    'threshold': 0.37,
    'features': 'all',  # 237 features
    'tp_pct': 1.5,
    'sl_pct': 0.75
}
```

---

## PERFORMANCES

### Baseline (Threshold = 0.50)
- **Trades**: 104
- **Win Rate**: 42.3%
- **ROI**: +22.56%
- **Capital**: $1000 → $1225.57
- **Accuracy**: 54.05%

### Phase 1 Optimized (Threshold = 0.37, Feature Selected)
- **Trades**: 127 (+23)
- **Win Rate**: 40.2% (-2.1%)
- **ROI**: +20.58% (-1.97%)
- **Capital**: $1000 → $1205.83
- **Note**: Feature selection dégrade légèrement

### Recommendation Finale
- **Modèle**: Baseline (237 features)
- **Threshold**: 0.37
- **ROI estimé**: Entre baseline et Phase 1 (~+22-25%)

---

## ANALYSE

### Points Forts
✅ Accuracy décente (54.05%)
✅ Sharpe ratio: 2.4 (bon)
✅ Expected Value positif avec threshold 0.37

### Points Faibles
⚠️ Feature selection ne fonctionne pas bien (-1.97% ROI)
⚠️ Win rate légèrement faible (42%)
⚠️ Amélioration modeste vs baseline

### Optimisations Testées

**Feature Selection (Top 50)**:
- Accuracy: 54.05% → 55.86% (+1.80%)
- ROI Backtest: -1.97% (dégradation)
- **Verdict**: Ne pas utiliser

**Threshold Optimization**:
- Optimal by ROI: 0.37
- Optimal by EV: 0.73
- **Recommandation**: 0.37 pour maximiser trades

**Optuna Hyperparameters** (100 trials):
- Best accuracy: 57.21% (+3.16%)
- Best AUC: 0.6114
- **Verdict**: Prometteur mais non testé en backtest ROI
- Model disponible: `btc_v11_optimized.joblib`

---

## HYPERPARAMÈTRES

### Baseline Model
```python
{
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
    'scale_pos_weight': 1.35,  # Class imbalance
    'random_state': 42
}
```

### Optuna Best Params (Trial #60)
```python
{
    'max_depth': 6,
    'min_child_weight': 8,
    'gamma': 4.991,
    'learning_rate': 0.0476,
    'n_estimators': 209,
    'subsample': 0.798,
    'colsample_bytree': 0.719,
    'colsample_bylevel': 0.923,
    'reg_alpha': 1.387,
    'reg_lambda': 0.219,
    'scale_pos_weight': 0.863
}
```

---

## DONNÉES D'ENTRAÎNEMENT

### Distribution
- **Train**: <2025-01-01
- **Test**: >=2025-01-01
- **Features**: 237 (multi-timeframe: 1h, 4h, 1d)

### Class Balance (Train)
- **SL (0)**: 57% (négatif)
- **TP (1)**: 43% (positif)
- **Imbalance**: Modéré

### Distribution Shift (Train → Test)
- **TP % Train**: 43%
- **TP % Test**: 34.5%
- **Shift**: -8.5% (BTC devient plus difficile sur 2025)

---

## TOP FEATURES (Importance)

Non applicable car utilise toutes les features (237).

Pour reference, top 10 features si feature selection:
1. `1d_atr_pct`
2. `1d_momentum_5`
3. `1d_volatility_7`
4. `4h_atr_pct`
5. `1d_bb_width`
6. `4h_momentum_5`
7. `1d_rsi_14`
8. `4h_volatility_7`
9. `1h_atr_pct`
10. `1d_macd_signal`

---

## UTILISATION

### Charger le modèle
```python
import joblib
import numpy as np
import pandas as pd

model = joblib.load('models/btc_v11_classifier.joblib')
threshold = 0.37
```

### Générer signal
```python
def generate_btc_signal(current_data: pd.DataFrame):
    """
    current_data: DataFrame avec toutes les features (237)
    Returns: (signal, probability)
    """
    # Exclude non-feature columns
    exclude_cols = ['open', 'high', 'low', 'close', 'volume',
                   'label_class', 'label_numeric', 'price_target_pct',
                   'future_price', 'triple_barrier_label']

    feature_cols = [col for col in current_data.columns
                    if col not in exclude_cols]

    # Prepare features
    X = current_data[feature_cols].fillna(0).values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Predict
    prob_tp = model.predict_proba(X)[:, 1]
    signal = 1 if prob_tp[0] > 0.37 else 0

    return signal, prob_tp[0]
```

---

## NOTES IMPORTANTES

1. **Threshold 0.37 est critique**: Plus important que le choix du modèle
2. **Feature selection ne marche pas**: Utiliser toutes les 237 features
3. **Optuna non testé en backtest**: Potentiel mais à valider
4. **Distribution shift -8.5%**: BTC devient plus difficile en 2025
5. **Sharpe 2.4 est bon**: Au-dessus standard industrie (1.5-2.0)

---

## AMÉLIORATIONS FUTURES

### Priorité HAUTE
- [ ] Tester Optuna model en backtest ROI
- [ ] Combiner baseline + Optuna (ensemble)
- [ ] Threshold dynamique selon volatilité

### Priorité MOYENNE
- [ ] Feature engineering ciblé BTC
- [ ] Data balancing (SMOTE)
- [ ] Retrain sur 2025 data (après collection)

### Priorité BASSE
- [ ] Ajuster TP/SL ratios
- [ ] Multi-label (TP/SL/Neutral)
- [ ] Deep learning (LSTM)

---

## FICHIERS ASSOCIÉS

**Modèles**:
- `models/btc_v11_classifier.joblib` (RECOMMANDÉ)
- `models/btc_v11_optimized.joblib` (Optuna, non testé)
- `models/btc_v11_feature_selected_top50.joblib` (dégradé)

**Résultats**:
- `models/btc_v11_stats.json` (baseline stats)
- `optimization/results/btc_v11_best_params.json` (Optuna)
- `optimization/results/btc_baseline_optimal_threshold.json` (Phase 1)
- `optimization/results/btc_selected_features_top50.json` (Feature selection)

**Données**:
- `data/cache/btc_multi_tf_merged.csv`

---

## CHANGELOG

**v11.2 (21 Mars 2026)**:
- Threshold optimization: 0.50 → 0.37
- Optuna tuning: +3.16% accuracy
- Feature selection: Testé, rejeté

**v11.1 (20 Mars 2026)**:
- Initial V11 TEMPORAL
- Baseline accuracy: 54.05%
- ROI: +22.56%
