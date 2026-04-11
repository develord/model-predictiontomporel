# FIL 08_train_production.py — Différences à corriger

## SHORT CNN params divergent du dev (03_train_short_model.py)

| Param | Dev (03_train_short) | Prod (08_train_production) | Action |
|-------|---------------------|---------------------------|--------|
| SHORT seq_length | **30** | 45 | ← corriger à 30 |
| SHORT LR | **0.001** | 0.0005 | ← corriger à 0.001 |
| SHORT noise | **0.02** | 0.01 | ← corriger à 0.02 |
| SHORT label_smooth | **0.05** | 0.1 | ← corriger à 0.05 |
| SHORT batch_size | **32** | 64 | ← corriger à 32 |

## Températures suspectes

- LONG=2.109, SHORT=2.157 — identiques à SOL et DOT
- Script probablement copié de SOL sans recalibrer
- **Action** : re-calibrer les températures sur un run dev FIL

## Meta XGBoost manquant

- Pas de `train_meta_xgboost()` dans le script
- FIL n'a même pas de meta models entraînés localement (models_meta/ vide)
- **Action** : d'abord entraîner meta en dev (05_train_meta_xgboost.py), puis ajouter au prod
